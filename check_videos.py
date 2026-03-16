#!/usr/bin/env python3
"""
check_videos.py — video integrity checker with optional auto-repair.

Scans video files using ffprobe/ffmpeg and reports container or decoding issues.
Targeted repair currently supports:
- non-monotonic DTS written to the muxer
- Opus packet-header parsing errors
- HEVC decode errors such as frame RPS/NALU corruption (re-encode repair)

Background:
I encountered these issues after editing videos with LosslessCut and enabling
Smart Cut, which sometimes produces files with timestamp problems and occasionally
other issues.

Authors: Me

Typical usage:
    check_videos.py PATH
    check_videos.py PATH --auto-fix
    check_videos.py --report report.json
    check_videos.py --repair-from-report report.json

For full CLI documentation, run:
    check_videos.py --help

History:
- v0.1 2026-03-14 Initial beta release.
- v0.2 2026-03-14 Added Opus packet-header auto-repair.
- v0.3 2026-03-14 Shortened CLI options (--auto-fix, --report).
- v0.4 2026-03-15 Reuse a single *.fixed target during fallback.
- v0.5 2026-03-15 Code cleanup and report wording normalization.
- v0.6 2026-03-16 Header and help reworked, added HEVC decode errors.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


VIDEO_EXTENSIONS = {
    ".mp4",
    ".m4v",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
    ".mts",
    ".m2ts",
    ".ts",
    ".wmv",
    ".flv",
    ".3gp",
    ".mpg",
    ".mpeg",
}

# Generic issue patterns from ffprobe/ffmpeg output.
ERROR_PATTERNS = (
    "error",
    "invalid",
    "corrupt",
    "missing",
    "non monotonically increasing dts",
    "non-monotonous dts",
    "moov atom not found",
    "truncat",
    "overread",
    "no frame",
    "broken",
)

WARN_PATTERNS = (
    "warning",
    "deprecated",
    "timestamp",
    "decode",
    "pts",
    "dts",
    "non-strictly-monotonic",
)

MUXER_DTS_PATTERNS = (
    "non monotonically increasing dts to muxer",
    "non-monotonous dts to muxer",
)

OPUS_PACKET_HEADER_PATTERNS = (
    "error parsing opus packet header",
)

HEVC_REPAIR_PATTERNS = (
    "error constructing the frame rps",
    "skipping invalid undecodable nalu",
)

MAX_AUTO_FIX_ATTEMPTS = 1


@dataclass
class CheckResult:
    path: Path
    checked_path: Path | None = None
    repaired_path: Path | None = None
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fix_logs: list[str] = field(default_factory=list)
    analysis: list[str] = field(default_factory=list)


@dataclass
class SourceProfile:
    video_codec: str | None = None
    audio_codec: str | None = None
    video_bitrate: int | None = None
    audio_bitrate: int | None = None
    format_bitrate: int | None = None


def parse_args() -> argparse.Namespace:
    blue_bold = "\033[1;34m"
    reset = "\033[0m"
    examples = f"""{blue_bold}Examples:{reset}
  check_videos.py
  check_videos.py clip.mp4
  check_videos.py DIR --report report.json
  check_videos.py clip.mp4 --auto-fix
  check_videos.py --repair-from-report report.json
"""
    description = (
        f"{blue_bold}Video integrity checker with targeted auto-repair.{reset}\n"
        "Scan video files using ffprobe/ffmpeg. Currently repairs DTS-to-muxer, Opus packet-header issues,\n"
        "and some HEVC decode errors."
    )
    parser = argparse.ArgumentParser(
        description=description,
        epilog=examples,
        formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
            prog, max_help_position=34, width=100
        ),
    )
    parser._positionals.title = "Positional Arguments"
    parser._optionals.title = "Options"
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="File or directory to scan recursively. Default: current directory.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use ffprobe only (faster, less thorough). Ignored with --repair-from-report.",
    )
    parser.add_argument(
        "--report",
        "--report-json",
        dest="report",
        default="",
        help="Write JSON results to this file.",
    )
    parser.add_argument(
        "--repair-from-report",
        metavar="REPORT",
        default="",
        help="Repair files listed in a JSON report instead of scanning.",
    )
    parser.add_argument(
        "--auto-fix",
        "--auto-fix-dts",
        dest="auto_fix",
        action="store_true",
        help=("Attempt repair for DTS-to-muxer and Opus packet-header issues."
              "Tries one remux first, then re-encodes if needed."),
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the original file instead of writing a .fixed file.",
    )
    parser.add_argument(
        "--deep-analyze-dts",
        action="store_true",
        help="Run packet-level DTS/PTS monotonicity analysis for video stream 0.",
    )
    return parser.parse_args()


def require_tool(name: str) -> None:
    if shutil.which(name):
        return
    print(
        f"Error: '{name}' not found. Please install ffmpeg/ffprobe.",
        file=sys.stderr,
    )
    raise SystemExit(2)


def iter_video_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, check=False)


def collect_pattern_lines(text: str, patterns: tuple[str, ...]) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        lower = line.lower()
        if any(p in lower for p in patterns):
            lines.append(line)
    return lines


def check_with_ffprobe(path: Path) -> tuple[list[str], list[str]]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-print_format",
        "json",
        str(path),
    ]
    proc = run_cmd(cmd)
    errors: list[str] = []
    warnings: list[str] = []

    if proc.returncode != 0:
        err = proc.stderr.strip() or "ffprobe return code != 0"
        errors.append(err)
        return errors, warnings

    if proc.stderr.strip():
        errors.extend(collect_pattern_lines(proc.stderr, ERROR_PATTERNS))
        warnings.extend(collect_pattern_lines(proc.stderr, WARN_PATTERNS))

    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        errors.append(f"ffprobe JSON invalid: {exc}")
        return errors, warnings

    streams = data.get("streams", [])
    if not streams:
        errors.append("No streams found.")
        return errors, warnings

    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    if not video_streams:
        errors.append("No video streams found.")

    for s in video_streams:
        idx = s.get("index", "?")
        if s.get("codec_name") in (None, "unknown"):
            errors.append(f"Stream {idx}: unknown codec.")
        width = s.get("width")
        height = s.get("height")
        if (width in (None, 0)) or (height in (None, 0)):
            warnings.append(f"Stream {idx}: suspicious resolution ({width}x{height}).")

    return errors, warnings


def check_with_ffmpeg_decode(path: Path) -> tuple[list[str], list[str]]:
    cmd = [
        "ffmpeg",
        "-v",
        "warning",
        "-i",
        str(path),
        "-map",
        "0:v:0?",
        "-map",
        "0:a?",
        "-sn",
        "-f",
        "null",
        "-",
    ]
    proc = run_cmd(cmd)
    text = "\n".join([proc.stdout or "", proc.stderr or ""]).strip()
    muxer_dts_warnings = collect_pattern_lines(text, MUXER_DTS_PATTERNS)
    errors = [
        line
        for line in collect_pattern_lines(text, ERROR_PATTERNS)
        if line not in muxer_dts_warnings
    ]
    warnings = collect_pattern_lines(text, WARN_PATTERNS)
    warnings.extend(muxer_dts_warnings)

    # Non-zero return code can still indicate runtime failure.
    if proc.returncode != 0 and not errors:
        errors.append(f"ffmpeg return code {proc.returncode}")
    return errors, warnings


def run_checks_for_path(path: Path, fast: bool) -> tuple[list[str], list[str], bool]:
    errors: list[str] = []
    warnings: list[str] = []

    probe_errors, probe_warnings = check_with_ffprobe(path)
    errors.extend(probe_errors)
    warnings.extend(probe_warnings)

    if not fast and not probe_errors:
        decode_errors, decode_warnings = check_with_ffmpeg_decode(path)
        errors.extend(decode_errors)
        warnings.extend(decode_warnings)

    errors = list(dict.fromkeys(errors))
    warnings = list(dict.fromkeys(warnings))
    return errors, warnings, len(errors) == 0


def has_repairable_issue(result: CheckResult) -> bool:
    all_lines = result.errors + result.warnings
    for line in all_lines:
        lower = line.lower()
        if any(p in lower for p in MUXER_DTS_PATTERNS):
            return True
        if any(p in lower for p in OPUS_PACKET_HEADER_PATTERNS):
            return True
        if any(p in lower for p in HEVC_REPAIR_PATTERNS):
            return True
    return False


def has_opus_packet_header_issue(result: CheckResult) -> bool:
    all_lines = result.errors + result.warnings
    for line in all_lines:
        lower = line.lower()
        if any(p in lower for p in OPUS_PACKET_HEADER_PATTERNS):
            return True
    return False


def has_hevc_repair_issue(result: CheckResult) -> bool:
    all_lines = result.errors + result.warnings
    for line in all_lines:
        lower = line.lower()
        if any(p in lower for p in HEVC_REPAIR_PATTERNS):
            return True
    return False


def next_fixed_path(source_path: Path) -> Path:
    candidate = source_path.with_name(f"{source_path.stem}.fixed{source_path.suffix}")
    if not candidate.exists():
        return candidate
    for idx in range(2, 1000):
        numbered = source_path.with_name(f"{source_path.stem}.fixed.{idx}{source_path.suffix}")
        if not numbered.exists():
            return numbered
    raise RuntimeError("Could not generate a free .fixed output filename.")


def replace_target_file(tmp_path: Path, target_path: Path) -> tuple[bool, str]:
    try:
        # Atomic-style replace; keep target until move succeeds.
        tmp_path.replace(target_path)
    except OSError as exc:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        return False, f"file replacement failed: {exc}"
    return True, ""


def remux_repair(input_path: Path, output_path: Path, attempt: int) -> tuple[bool, str]:
    tmp_name = f".{output_path.stem}.dtsfix.{os.getpid()}.{attempt}{output_path.suffix}"
    tmp_path = output_path.with_name(tmp_name)
    cmd = [
        "ffmpeg",
        "-v",
        "warning",
        "-fflags",
        "+genpts",
        "-i",
        str(input_path),
        "-map",
        "0",
        "-c",
        "copy",
    ]
    if output_path.suffix.lower() in {".mp4", ".m4v", ".mov"}:
        cmd.extend(["-movflags", "+faststart"])
    cmd.extend(["-y", str(tmp_path)])

    proc = run_cmd(cmd)
    text = "\n".join([proc.stdout or "", proc.stderr or ""]).strip()
    if proc.returncode != 0:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        details = collect_pattern_lines(text, ERROR_PATTERNS)[:1]
        tail = f": {details[0]}" if details else ""
        return False, f"auto-fix attempt {attempt} failed (ffmpeg rc={proc.returncode}{tail})"

    if not tmp_path.exists() or tmp_path.stat().st_size == 0:
        tmp_path.unlink(missing_ok=True)
        return False, f"auto-fix attempt {attempt} failed (empty output file)"

    switched, switch_err = replace_target_file(tmp_path, output_path)
    if not switched:
        return False, f"auto-fix attempt {attempt} failed ({switch_err})"

    return (
        True,
        f"auto-fix attempt {attempt}: remux complete ({input_path.name} -> {output_path.name}), recheck follows.",
    )


def _bitrate_arg(value: int | None, fallback: str) -> str:
    if not value or value <= 0:
        return fallback
    return str(value)


def pick_video_encoder(profile: SourceProfile) -> tuple[list[str], str]:
    codec = (profile.video_codec or "").lower()
    source_vbr = profile.video_bitrate

    if codec == "h264":
        args = ["libx264", "-crf", "20", "-preset", "slow"]
        if source_vbr:
            cap = max(int(source_vbr * 0.95), 300_000)
            args.extend(["-maxrate", _bitrate_arg(cap, "1200000"), "-bufsize", _bitrate_arg(cap * 2, "2400000")])
            return args, f"h264 -> libx264 (maxrate~{cap}bps)"
        return args, "h264 -> libx264 (CRF 20)"
    if codec in {"hevc", "h265"}:
        args = ["libx265", "-crf", "22", "-preset", "slow"]
        if source_vbr:
            cap = max(int(source_vbr * 0.95), 250_000)
            args.extend(["-maxrate", _bitrate_arg(cap, "1000000"), "-bufsize", _bitrate_arg(cap * 2, "2000000")])
            return args, f"hevc -> libx265 (maxrate~{cap}bps)"
        return args, "hevc -> libx265 (CRF 22)"
    if codec == "mpeg4":
        return ["mpeg4", "-q:v", "3"], "mpeg4 -> mpeg4 (q:v 3)"
    if codec == "mpeg2video":
        return ["mpeg2video", "-q:v", "3"], "mpeg2video -> mpeg2video (q:v 3)"
    if codec == "vp9":
        if source_vbr:
            target = max(int(source_vbr * 0.9), 250_000)
            return ["libvpx-vp9", "-crf", "34", "-b:v", _bitrate_arg(target, "800000")], f"vp9 -> libvpx-vp9 ({target}bps)"
        return ["libvpx-vp9", "-crf", "34", "-b:v", "0"], "vp9 -> libvpx-vp9 (CRF 34)"
    if codec == "av1":
        if source_vbr:
            target = max(int(source_vbr * 0.9), 200_000)
            return ["libaom-av1", "-crf", "32", "-b:v", _bitrate_arg(target, "700000")], f"av1 -> libaom-av1 ({target}bps)"
        return ["libaom-av1", "-crf", "32", "-b:v", "0"], "av1 -> libaom-av1 (CRF 32)"

    args = ["libx264", "-crf", "21", "-preset", "slow"]
    if source_vbr:
        cap = max(int(source_vbr * 0.95), 300_000)
        args.extend(["-maxrate", _bitrate_arg(cap, "1200000"), "-bufsize", _bitrate_arg(cap * 2, "2400000")])
        return args, f"{codec or 'unknown'} -> libx264 fallback (maxrate~{cap}bps)"
    return args, f"{codec or 'unknown'} -> libx264 fallback (CRF 21)"


def pick_audio_encoder(
    profile: SourceProfile,
    *,
    output_suffix: str,
    prefer_non_opus: bool = False,
) -> tuple[list[str], str]:
    codec = (profile.audio_codec or "").lower()
    source_abr = profile.audio_bitrate
    aac_br = min(max(source_abr or 128_000, 96_000), 192_000)

    if codec == "aac":
        return ["aac", "-b:a", _bitrate_arg(aac_br, "128000")], f"aac -> aac ({aac_br}bps)"
    if codec == "mp3":
        return ["libmp3lame", "-q:a", "2"], "mp3 -> libmp3lame"
    if codec == "ac3":
        ac3_br = min(max(source_abr or 384_000, 192_000), 448_000)
        return ["ac3", "-b:a", _bitrate_arg(ac3_br, "384000")], f"ac3 -> ac3 ({ac3_br}bps)"
    if codec == "eac3":
        eac3_br = min(max(source_abr or 256_000, 128_000), 448_000)
        return ["eac3", "-b:a", _bitrate_arg(eac3_br, "256000")], f"eac3 -> eac3 ({eac3_br}bps)"
    if codec == "opus":
        if prefer_non_opus:
            if output_suffix in {".webm", ".mkv"}:
                return ["libvorbis", "-q:a", "5"], "opus -> libvorbis (robust repair mode)"
            return ["aac", "-b:a", _bitrate_arg(aac_br, "128000")], "opus -> aac (robust repair mode)"
        opus_br = min(max(source_abr or 128_000, 96_000), 192_000)
        return ["libopus", "-b:a", _bitrate_arg(opus_br, "128000")], f"opus -> libopus ({opus_br}bps)"
    if codec == "vorbis":
        return ["libvorbis", "-q:a", "5"], "vorbis -> libvorbis"
    if codec == "flac":
        return ["flac"], "flac -> flac"
    return ["aac", "-b:a", _bitrate_arg(aac_br, "128000")], f"{codec or 'unknown'} -> aac fallback ({aac_br}bps)"


def read_source_profile(path: Path) -> tuple[SourceProfile, str | None]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_format",
        "-show_streams",
        "-print_format",
        "json",
        str(path),
    ]
    proc = run_cmd(cmd)
    if proc.returncode != 0:
        return SourceProfile(), proc.stderr.strip() or f"ffprobe rc={proc.returncode}"
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        return SourceProfile(), f"ffprobe JSON invalid: {exc}"

    streams = data.get("streams", [])
    v_stream = next(
        (s for s in streams if s.get("codec_type") == "video"),
        None,
    )
    a_stream = next(
        (s for s in streams if s.get("codec_type") == "audio"),
        None,
    )
    fmt = data.get("format", {}) if isinstance(data.get("format"), dict) else {}

    def parse_int(value: object) -> int | None:
        try:
            if value in (None, "", "N/A"):
                return None
            return int(str(value))
        except (TypeError, ValueError):
            return None

    profile = SourceProfile(
        video_codec=v_stream.get("codec_name") if v_stream else None,
        audio_codec=a_stream.get("codec_name") if a_stream else None,
        video_bitrate=parse_int(v_stream.get("bit_rate")) if v_stream else None,
        audio_bitrate=parse_int(a_stream.get("bit_rate")) if a_stream else None,
        format_bitrate=parse_int(fmt.get("bit_rate")),
    )
    return profile, None


def reencode_keep_codec_family(
    input_path: Path,
    output_path: Path,
    *,
    prefer_non_opus_audio: bool = False,
) -> tuple[bool, list[str]]:
    profile, read_err = read_source_profile(input_path)
    if read_err:
        return False, [f"Re-encode fallback could not read source profile: {read_err}"]

    if profile.video_bitrate is None and profile.format_bitrate:
        estimated_audio = profile.audio_bitrate or 128_000
        profile.video_bitrate = max(profile.format_bitrate - estimated_audio, 250_000)

    v_args, v_note = pick_video_encoder(profile)
    a_args, a_note = pick_audio_encoder(
        profile,
        output_suffix=output_path.suffix.lower(),
        prefer_non_opus=prefer_non_opus_audio,
    )
    tmp_name = f".{output_path.stem}.dtsfix.reencode.{os.getpid()}{output_path.suffix}"
    tmp_path = output_path.with_name(tmp_name)
    logs = [
        "Starting re-encode fallback (source codec families + size-aware presets).",
        f"source: {input_path.name}",
        f"target: {output_path.name}",
        f"source bitrate total: {profile.format_bitrate or 'unknown'} bps",
        f"source bitrate video: {profile.video_bitrate or 'unknown'} bps",
        f"source bitrate audio: {profile.audio_bitrate or 'unknown'} bps",
        f"Video: {v_note}",
        f"Audio: {a_note}",
    ]

    cmd = [
        "ffmpeg",
        "-v",
        "warning",
        "-i",
        str(input_path),
        "-map",
        "0",
        "-map",
        "-0:d?",
        "-map",
        "-0:t?",
        "-c:v",
        *v_args,
        "-c:a",
        *a_args,
        "-c:s",
        "copy",
    ]
    if output_path.suffix.lower() in {".mp4", ".m4v", ".mov"}:
        cmd.extend(["-movflags", "+faststart"])
    cmd.extend(["-y", str(tmp_path)])

    proc = run_cmd(cmd)
    text = "\n".join([proc.stdout or "", proc.stderr or ""]).strip()
    if proc.returncode != 0:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        details = collect_pattern_lines(text, ERROR_PATTERNS)[:1]
        tail = f": {details[0]}" if details else ""
        logs.append(f"re-encode fallback failed (ffmpeg rc={proc.returncode}{tail})")
        return False, logs

    if not tmp_path.exists() or tmp_path.stat().st_size == 0:
        tmp_path.unlink(missing_ok=True)
        logs.append("re-encode fallback failed (empty output file)")
        return False, logs

    switched, switch_err = replace_target_file(tmp_path, output_path)
    if not switched:
        logs.append(f"re-encode fallback failed ({switch_err})")
        return False, logs

    logs.append("re-encode fallback complete, recheck follows.")
    return True, logs


def analyze_dts_timeline(path: Path, max_examples: int = 5) -> list[str]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_packets",
        "-show_entries",
        "packet=dts,dts_time,pts,pts_time,flags",
        "-print_format",
        "json",
        str(path),
    ]
    proc = run_cmd(cmd)
    if proc.returncode != 0:
        err = proc.stderr.strip() or f"ffprobe return code {proc.returncode}"
        return [f"DTS analysis failed: {err}"]

    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        return [f"DTS analysis JSON invalid: {exc}"]

    packets = data.get("packets", [])
    if not packets:
        return ["DTS analysis: no packets found in video stream 0."]

    issues: list[str] = []
    prev_dts: int | None = None
    prev_time = "?"
    checked = 0

    for idx, pkt in enumerate(packets):
        dts_raw = pkt.get("dts")
        if dts_raw in (None, "N/A"):
            continue
        try:
            dts = int(dts_raw)
        except (TypeError, ValueError):
            continue
        dts_time = str(pkt.get("dts_time", "?"))
        if prev_dts is not None and dts <= prev_dts:
            issues.append(
                f"DTS not monotonic at packet {idx}: {prev_dts} ({prev_time}) >= {dts} ({dts_time})"
            )
            if len(issues) >= max_examples:
                break
        prev_dts = dts
        prev_time = dts_time
        checked += 1

    summary = f"DTS analysis: checked {checked} packets with DTS."
    if not issues:
        return [summary, "DTS analysis: no non-monotonic DTS found in v:0."]
    return [summary, f"DTS analysis: found {len(issues)} example DTS jumps."] + issues


def check_file(
    path: Path,
    fast: bool,
    auto_fix: bool,
    deep_analyze_dts: bool,
    inplace: bool,
) -> CheckResult:
    result = CheckResult(path=path)
    work_path = path
    saw_opus_packet_header_issue = False

    for attempt in range(0, MAX_AUTO_FIX_ATTEMPTS + 1):
        result.errors, result.warnings, result.ok = run_checks_for_path(work_path, fast)

        issue_still_present = has_repairable_issue(result)
        saw_opus_packet_header_issue = saw_opus_packet_header_issue or has_opus_packet_header_issue(result)
        saw_hevc_repair_issue = has_hevc_repair_issue(result)
        if fast or not auto_fix or not issue_still_present:
            if auto_fix and attempt > 0:
                result.fix_logs.append(f"auto-fix successful after attempt {attempt}.")
            break
        if saw_hevc_repair_issue:
            result.fix_logs.append("auto-fix: HEVC decode issue detected, using re-encode repair.")
            fallback_target = work_path if (inplace or work_path != path) else next_fixed_path(path)
            reencode_ok, reencode_logs = reencode_keep_codec_family(
                work_path,
                fallback_target,
                prefer_non_opus_audio=saw_opus_packet_header_issue,
            )
            result.fix_logs.extend(reencode_logs)
            if reencode_ok:
                work_path = fallback_target
                result.fix_logs.append("recheck after HEVC re-encode repair.")
                result.errors, result.warnings, result.ok = run_checks_for_path(work_path, fast)
            break
        if attempt >= MAX_AUTO_FIX_ATTEMPTS:
            result.fix_logs.append(
                "auto-fix ended: issue still present after "
                f"{MAX_AUTO_FIX_ATTEMPTS} attempts."
            )
            # Keep a single repaired output file in non-inplace mode:
            # if remux already produced *.fixed, overwrite that same file in fallback.
            fallback_target = work_path if (inplace or work_path != path) else next_fixed_path(path)
            reencode_ok, reencode_logs = reencode_keep_codec_family(
                work_path,
                fallback_target,
                prefer_non_opus_audio=saw_opus_packet_header_issue,
            )
            result.fix_logs.extend(reencode_logs)
            if reencode_ok:
                work_path = fallback_target
                result.fix_logs.append("recheck after re-encode fallback.")
                result.errors, result.warnings, result.ok = run_checks_for_path(work_path, fast)
            break

        remux_target = work_path if inplace or work_path != path else next_fixed_path(path)
        ok, log_line = remux_repair(work_path, remux_target, attempt + 1)
        result.fix_logs.append(log_line)
        if ok:
            work_path = remux_target
        if not ok:
            result.fix_logs.append(
                f"auto-fix attempt {attempt + 1}: remux failed, trying next step."
            )

    result.checked_path = work_path
    if work_path != path:
        result.repaired_path = work_path
        result.fix_logs.append(f"original unchanged: {path.name}")
        result.fix_logs.append(f"repaired file: {work_path.name}")

    if deep_analyze_dts:
        result.analysis = analyze_dts_timeline(work_path)

    return result


def print_result_line(index: int, total: int, result: CheckResult) -> None:
    rel = result.path.as_posix()
    if result.ok and not result.warnings:
        print(f"[{index}/{total}] OK       {rel}")
        if result.repaired_path:
            print(f"    - source: {result.path.name}")
            print(f"    - checked: {result.checked_path.name if result.checked_path else result.path.name}")
            print(f"    - repaired: {result.repaired_path.name}")
        return
    if result.ok and result.warnings:
        print(f"[{index}/{total}] WARNING  {rel}")
        if result.checked_path and result.checked_path != result.path:
            print(f"    - checked: {result.checked_path.name}")
        for line in result.fix_logs:
            print(f"    - {line}")
        for line in result.analysis:
            print(f"    - {line}")
        for w in result.warnings[:3]:
            print(f"    - {w}")
        if len(result.warnings) > 3:
            print(f"    - ... {len(result.warnings) - 3} more warnings")
        return
    print(f"[{index}/{total}] ERROR    {rel}")
    if result.checked_path and result.checked_path != result.path:
        print(f"    - checked: {result.checked_path.name}")
    for line in result.fix_logs:
        print(f"    - {line}")
    for line in result.analysis:
        print(f"    - {line}")
    for e in result.errors[:5]:
        print(f"    - {e}")
    if len(result.errors) > 5:
        print(f"    - ... {len(result.errors) - 5} more errors")


def write_json_report(path: Path, results: list[CheckResult]) -> None:
    payload = {
        "checked_files": len(results),
        "ok_files": sum(1 for r in results if r.ok),
        "error_files": sum(1 for r in results if not r.ok),
        "warning_files": sum(1 for r in results if r.warnings),
        "results": [
            {
                "path": str(r.path),
                "ok": r.ok,
                "checked_path": str(r.checked_path) if r.checked_path else "",
                "repaired_path": str(r.repaired_path) if r.repaired_path else "",
                "errors": r.errors,
                "warnings": r.warnings,
                "fix_logs": r.fix_logs,
                "analysis": r.analysis,
            }
            for r in results
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def load_problem_files_from_report(report_path: Path) -> tuple[list[tuple[Path, str]], str | None]:
    def resolve_report_file_path(value: object) -> Path | None:
        if not isinstance(value, str) or not value.strip():
            return None
        p = Path(value).expanduser()
        if not p.is_absolute():
            p = (report_path.parent / p).resolve()
        return p

    try:
        raw = report_path.read_text(encoding="utf-8")
    except OSError as exc:
        return [], f"cannot read report file: {exc}"

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return [], f"invalid report JSON: {exc}"

    items = data.get("results")
    if not isinstance(items, list):
        return [], "report does not contain a valid 'results' list."

    files: list[tuple[Path, str]] = []
    # Per-path reason; prefer "error" over "warning" for duplicates.
    reason_by_path: dict[str, str] = {}
    path_by_key: dict[str, Path] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        errors = item.get("errors") or []
        warnings = item.get("warnings") or []
        if not errors and not warnings:
            continue
        reason = "error" if errors else "warning"

        candidates = [
            resolve_report_file_path(item.get("repaired_path")),
            resolve_report_file_path(item.get("checked_path")),
            resolve_report_file_path(item.get("path")),
        ]
        existing = next((p for p in candidates if p and p.exists()), None)
        target = existing or next((p for p in candidates if p), None)
        if target is None:
            continue

        key = str(target)
        path_by_key[key] = target
        prev_reason = reason_by_path.get(key)
        if prev_reason is None or (prev_reason == "warning" and reason == "error"):
            reason_by_path[key] = reason

    for key, reason in reason_by_path.items():
        files.append((path_by_key[key], reason))
    return files, None


def print_problem_file_list(results: list[CheckResult]) -> None:
    problem_results = [r for r in results if r.errors or r.warnings]
    if not problem_results:
        print("- Problem files: none")
        return

    print(f"- Problem files: {len(problem_results)}")
    for r in problem_results:
        status = "error" if r.errors else "warning"
        print(f"  - {r.path} ({status})")


def main() -> int:
    args = parse_args()
    start_path = Path(args.path).expanduser()
    repair_from_report = Path(args.repair_from_report).expanduser() if args.repair_from_report else None
    auto_fix_enabled = args.auto_fix or bool(repair_from_report)
    repair_reason_by_path: dict[str, str] = {}

    if repair_from_report:
        if not repair_from_report.exists():
            print(f"error: report file not found: {repair_from_report}", file=sys.stderr)
            return 2
        report_targets, load_err = load_problem_files_from_report(repair_from_report)
        if load_err:
            print(f"error: {load_err}", file=sys.stderr)
            return 2
        if not report_targets:
            print(f"No problem files found in report: {repair_from_report}")
            return 0
        files = [p for p, _ in report_targets]
        repair_reason_by_path = {str(p): reason for p, reason in report_targets}
    else:
        if not start_path.exists():
            print(f"Error: path not found: {start_path}", file=sys.stderr)
            return 2
        if start_path.is_file():
            if start_path.suffix.lower() not in VIDEO_EXTENSIONS:
                print(f"Unsupported video file: {start_path}", file=sys.stderr)
                return 2
            files = [start_path]
        else:
            files = sorted(iter_video_files(start_path))
        if not files:
            print(f"No video files found under: {start_path}")
            return 0

    require_tool("ffprobe")
    if auto_fix_enabled or not args.fast:
        require_tool("ffmpeg")

    effective_fast = args.fast
    if repair_from_report and effective_fast:
        print("Note: --fast is disabled in repair mode so repair checks can run.")
        effective_fast = False

    if repair_from_report:
        print(f"Repair run from report: {repair_from_report}")
        print(f"Files loaded for repair: {len(files)}")
        warning_only = sum(1 for _, reason in repair_reason_by_path.items() if reason == "warning")
        if warning_only:
            print(
                f"Note: {warning_only} warning-only file(s) included from report "
                "(best-effort repair attempt)."
            )
    else:
        print(f"Scanning {len(files)} video files under: {start_path}")
    print(f"Mode: {'fast (ffprobe)' if effective_fast else 'thorough (ffprobe+ffmpeg)'}")
    if auto_fix_enabled:
        print(
            "Auto-fix: enabled (max "
            f"{MAX_AUTO_FIX_ATTEMPTS} attempt(s) for repairable mux/decode issues)"
        )
        print(f"Auto-fix output: {'overwrite original file' if args.inplace else 'write a new .fixed file'}")
    if args.deep_analyze_dts:
        print("DTS analysis: enabled (packet-level ffprobe for v:0)")

    results: list[CheckResult] = []
    for i, file_path in enumerate(files, start=1):
        if repair_from_report and repair_reason_by_path.get(str(file_path)) == "warning":
            print(f"[{i}/{len(files)}] INFO     {file_path.as_posix()}")
            print("    - warning-only report entry; running best-effort repair check.")
        if not file_path.exists():
            missing = CheckResult(path=file_path, ok=False, errors=["file not found"])
            results.append(missing)
            print_result_line(i, len(files), missing)
            continue
        result = check_file(
            file_path,
            fast=effective_fast,
            auto_fix=auto_fix_enabled,
            deep_analyze_dts=args.deep_analyze_dts,
            inplace=args.inplace,
        )
        results.append(result)
        print_result_line(i, len(files), result)

    error_count = sum(1 for r in results if not r.ok)
    warn_count = sum(1 for r in results if r.warnings)
    print("\nSummary")
    print(f"- Checked:  {len(results)}")
    print(f"- Errors:   {error_count}")
    print(f"- Warnings: {warn_count}")
    print_problem_file_list(results)

    if args.report:
        report_path = Path(args.report).expanduser()
        write_json_report(report_path, results)
        print(f"- Report: {report_path}")
        if not repair_from_report:
            print(f"- Repair command: ./check_videos.py --repair-from-report {report_path}")

    return 1 if error_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
