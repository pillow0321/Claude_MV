# Reflection_Eter — Music Video

This is a music video for **Reflection_Eter** by wanderingnonet286. It is generated entirely from code and the audio file, with no stock footage.

- `Reflection_Eter_MV.mp4`: the finished video (1920x1080, 24fps, 2.39:1 letterbox, 3:33)
- `mv.py`: the renderer. It analyses the audio (BPM, kicks, spectrum) and draws every frame procedurally.
- `GOAL.md`: the goal and the 100-point self-scoring rubric
- `SCORE.md`: the self-scoring result

## Concept

The visuals take the palette and motifs from the single's cover art: blue-grey monochrome, a film strip, a poppy silhouette, raindrops, clouds, and one line of light across the frame.

- **Reflection**: a mirror-still lake. The horizon line of light runs through the whole video and ties the chapters together. Its halo moves with the song's live spectrum.
- **Eter (ether)**: in the finale, particles of light rise from the water into the sky.

| Time | Chapter | Scene |
|---|---|---|
| 0:00 | I. mist | Mist, the line of light spreads out, title |
| 0:21 | II. surface | Moon over a mirror lake; ripples on the beat. The screen darkens with the silence before the drop |
| 1:02 | III. bloom | Hard cut on the drop. Poppy silhouette in rain, swaying with the kick |
| 1:44 | IV. frames | Film strip taken from the cover art, scrolling one frame every 2 bars |
| 2:24 | V. stillness | Break. Every 2 bars a single drop meets its own reflection on the downbeat |
| 2:47 | VI. ether | Climax. Aurora, rising particles, ripples on every beat |
| 3:23 | VII. reflection | Afterglow and end card |

## Re-rendering

```bash
pip install numpy scipy pillow imageio-ffmpeg
python3 mv.py Reflection_Eter.mp3 master.mp4             # full render (about 12 minutes on 4 cores)
python3 mv.py Reflection_Eter.mp3 stills --stills 40,90  # check stills
```

Fonts: Instrument Serif, DM Mono and Jura (all SIL Open Font License).
