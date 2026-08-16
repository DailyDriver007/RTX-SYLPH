# SYLPH animation assets

Clips are grouped by SYLPH's states of being:

| State | Filename prefix | Used when |
|---|---|---|
| `idle` | `SYLPH_IDLE_*.mp4`, `SYLPH_Idle.mp4`, `rtx_sylph_animated.mp4` | Resting HUD |
| `listening` | same as idle + blue tint | Wake word heard |
| `thinking` | `SYLPH_THINKING_*.mp4` | AI council is querying |
| `answering` | `SYLPH_ANSWERS_*.mp4` | Results are on screen |
| `gpu_cool` | `GPU_COOL_*.mp4` | GPU load is low |
| `gpu_overheat` | `GPU_OVERHEAT_*.mp4` | GPU temp is high |
| `home_assist` | `SYLPH_HOME_ASSIST_*.mp4` | Lights / climate / home commands |
| `sound_system` | `RTX_SYLPH_Sound_System*.mp4` | Music / media |
| `camera_mode` | `SYLPH_CAMERA_*.mp4`, `SYLPH_PC_ROG.mp4` | Camera spiral / view |

Public pack (all clips):

https://drive.google.com/drive/folders/1FCswEHSVKXkJmv4drEToIs0MBZLzRltX?usp=sharing

Drop the mp4 files into `assets/` or `assets/assets/`. The plugin discovers them by prefix, so missing numbers are fine.
