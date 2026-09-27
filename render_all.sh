#!/usr/bin/env bash
# Render the MV in 30 s segments (resumable), then join and mux the audio.
set -euo pipefail
AUDIO=${1:?audio}; OUT=${2:?out.mp4}; SEG=${SEG_DIR:-segments}
mkdir -p "$SEG"
FF=$(python3 -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
: > "$SEG/list.txt"
for s in $(seq 0 30 210); do
  e=$((s+30)); f="$SEG/seg_$(printf %03d $s).mp4"
  [ -s "$f" ] || python3 "$(dirname "$0")/mv.py" "$AUDIO" "$f" --start $s --end $e --video-only
  echo "file '$(realpath "$f")'" >> "$SEG/list.txt"
done
"$FF" -y -v error -f concat -safe 0 -i "$SEG/list.txt" -i "$AUDIO" -map 0:v -map 1:a \
  -c:v copy -c:a aac -b:a 320k -movflags +faststart -shortest "$OUT"
echo "done $OUT"
