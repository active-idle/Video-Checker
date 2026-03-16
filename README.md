# Check Videos
Video integrity checker with optional auto-repair.

Scans video files using ffprobe/ffmpeg and reports container or decoding issues.
Targeted repair currently supports:
- non-monotonic DTS written to the muxer
- Opus packet-header parsing errors
- HEVC decode errors such as frame RPS/NALU corruption (re-encode repair)

## Background
I encountered these issues after editing videos with LosslessCut and enabling
Smart Cut, which sometimes produces files with timestamp problems and occasionally
other issues.

## Usage and Details
Run the script using Python 3.
See `check_videos.py --help` for options and examples.

## Donation
Thank you very much for a donation in recognition of my work --> [![Paypal Donate](https://img.shields.io/badge/paypal-donate-yellow.svg)](https://www.paypal.com/paypalme/MrDagoo/)
