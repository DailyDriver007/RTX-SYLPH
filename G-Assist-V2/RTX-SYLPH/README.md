# RTX SYLPH — G-Assist V2 plugin

Voice-activated companion for [NVIDIA Project G-Assist](https://github.com/NVIDIA/g-assist).  
States of being drive her animation classes. The AI council opens one window per model.

## States of being

| State | Animation prefix | Trigger |
|---|---|---|
| idle | `SYLPH_IDLE_*` | Resting |
| listening | idle + blue tint | Wake word **sylph** |
| thinking | `SYLPH_THINKING_*` | Council is querying |
| answering | `SYLPH_ANSWERS_*` | Results on screen |
| gpu_cool | `GPU_COOL_*` | Low GPU load |
| gpu_overheat | `GPU_OVERHEAT_*` | High GPU temp |
| home_assist | `SYLPH_HOME_ASSIST_*` | Home Assistant commands |
| sound_system | `RTX_SYLPH_Sound_System*` | Music / media |
| camera_mode | `SYLPH_CAMERA_*` / `SYLPH_PC_ROG` | Cameras |

Public clip pack (do not commit the mp4s here):

https://drive.google.com/drive/folders/1FCswEHSVKXkJmv4drEToIs0MBZLzRltX?usp=sharing

Copy the videos into `assets/` or `assets/assets/`.

## Install (G-Assist)

1. Python **3.10** (the plugin uses OpenCV, pygame, PyAudio).
2. `py -3.10 -m pip install -r requirements.txt`
3. Copy `config.example.json` to `config.json` and add **your** keys locally. Never commit `config.json`.
4. Copy this folder to:

   `%PROGRAMDATA%\NVIDIA Corporation\nvtopps\rise\plugins\RTX-SYLPH`

5. Restart G-Assist.

Standalone HUD:

```bat
run_sylph.bat
```

## Commands

`ask_ai`, `gpu_status`, `lights_control`, `thermostat_control`, `home_control`, `camera_spiral`, `camera_view`, `find_airtag`, `screencast`, `close_all`, plus avatar controls.

`ask_ai` writes one HTML file per model and opens it in a positioned Chrome/Edge app window (inner ring = originals, offset ring = Grok-led refinements).

## Secrets

API keys stay in local `config.json` only. This repo ships empty placeholders in `config.example.json`.
