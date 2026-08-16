# RTX-SYLPH

RTX SYLPH is an ethereal blue-to-NVIDIA-neon-green sprite. She sits as an upper-left HUD, listens for **sylph**, and runs a multi-model AI council (Grok, ChatGPT, Nemotron, Llama, DeepInfra, Mistral) plus GPU, home, and camera commands.

## Layout

| Path | What |
|---|---|
| `SYLPH.py` | Original standalone companion |
| `RTXSYLPH/G-Assist/` | v7.7 standalone / installer sources |
| `G-Assist-V2/RTX-SYLPH/` | **Current NVIDIA Project G-Assist V2 plugin** |

## G-Assist V2 (current)

See [`G-Assist-V2/RTX-SYLPH/README.md`](G-Assist-V2/RTX-SYLPH/README.md).

1. Install Python 3.10 + `requirements.txt`
2. Copy `config.example.json` → `config.json` and add your own keys (never commit them)
3. Drop animation clips from the public pack into `assets/`
4. Copy the plugin folder into `%PROGRAMDATA%\NVIDIA Corporation\nvtopps\rise\plugins\RTX-SYLPH`

Animation pack:

https://drive.google.com/drive/folders/1FCswEHSVKXkJmv4drEToIs0MBZLzRltX?usp=sharing

mod.io listing (manual upload): https://mod.io/g/g-assist/m/rtxsylph  
Notes: [`MODIO.md`](MODIO.md)

## States of being

idle · listening · thinking · answering · gpu_cool · gpu_overheat · home_assist · sound_system · camera_mode

Each state plays the matching Drive clip class. The council opens one real browser window per model (HTML files, not data-URLs).
