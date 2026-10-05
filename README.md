# RTX-SYLPH

RTX SYLPH is an ethereal blue-to-NVIDIA-neon-green sprite. She sits as a centered HUD, listens for **sylph**, and runs a multi-model AI council (Grok, Gemini Flash 3.7, Gemini 3.1 Pro, Nemotron 3.5 Lightning, Nemotron 3 Super, Nemotron Ultra 253B, Mistral) plus GPU, home, camera, Ara voice, and a world clock.

## License

Copyright (c) 2025–2026 Kyle Baker. All rights reserved. The terms are in [`LICENSE`](LICENSE).

You may download one copy for your own private, non-commercial use. That copy is not for public display. You may not use this code in a commercial project, copy it into another product, repository, or service, or re-skin SYLPH unless Kyle Baker grants that right in a written license. A download, fork, or mod.io subscription is not that grant.

Commercial terms, when a written license is granted: Kyle Baker retains the copyright and the SYLPH name. The license is non-exclusive, non-transferable, and cannot be sublicensed. The customer may run the official build on the machines named in the grant. The customer may not copy the code into another product, repository, or service. The license ends if they do.

License requests: youshould@getacustom.one

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
