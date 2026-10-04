# RTX SYLPH — G-Assist V2 plugin

Voice-activated companion for [NVIDIA Project G-Assist](https://github.com/NVIDIA/g-assist).  
States of being drive her animation classes. The AI council opens one window per model.

Council seats: **Grok**, **Gemini Flash 3.7**, **Gemini 3.1 Pro**, **Nemotron 3.5 Lightning**, **Nemotron 3 Super**, **Nemotron Ultra 253B**, **Mistral**.  
DeepInfra and Llama seats are parked for local use. NVIDIA models use the free NIM endpoint (`integrate.api.nvidia.com`). Voice prefers **Ara** through xAI TTS. An always-on world clock sits on the desktop, and a text console on the left accepts typed messages, files, images, video, and links.

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

Say **sylph** then a full sentence — the mic waits through conversational pauses (about 1.5s of silence) so it does not clip you. Ask **what time** for a spoken world-clock snapshot.

**SYLPH DESK** (compact tiles, starts top-center): GPU sparkline, weather+AQI, calendar (ICS URL), focus timer, now-playing mixer, clipboard vault, Home (HA / Google Home / Alexa), link shelf, quiet hours, screenshot→ask, Grok Bot, PRO/Flight. Drag the **☰ SYLPH DESK** handle to park it anywhere — position is remembered. Double-click the handle or say **reset desk** to snap it back to the top. Click a tile to expand. HA/GGL/ALX picks the smart-home environment. Live HA tiles need `HA_URL` + `HA_KEY`. Calendar needs a Google Calendar **secret iCal** URL in `CALENDAR_ICS_URL`. **BOT** opens official Grok Bot (`https://x.ai/bot`) — included with SuperGrok Plus; there is no public Bot API yet.

**Studio vs Flight:** she ships as the witty truncated **studio** edition (rehearsal). A license key (`unlock SYLPH-PREM-…`) turns on **Flight**: full xAI Grok, truth-seeking prompts, longer spoken answers, model thinking, and a local NVIDIA hardware bible (5090 / Blackwell / GB200). This machine can also `SYLPH_OWNER_FLIGHT: true` in local `config.json` to mint a perpetual Flight unlock automatically (do not ship that flag as true). Licensed users can **demo studio** and drag **accuracy / wit / depth / spoken** on the PRO desk tile, or say `flight mode`, `demo studio`, `accuracy 90`, `increase wit`. Pricing: $50 perpetual or $8/month — see `MARKETPLACES.md`. Issue keys with `issue_license.py`.

Packaging and where to sell: see `MARKETPLACES.md`.

The **SYLPH CONSOLE** (left of the HUD) is a text window: type, **Attach** jpg/png/mp4/files, **Paste** a clipboard image, or drop a URL in the message. **Send** talks to her; **Council** also opens the model ring. Ctrl+Enter sends.

**Watch desk** (bottom-right): say **open Netflix**, **open Prime**, **what's new on Netflix**, or **what should I watch**. She launches *your* browser/app and reads each TMDB synopsis out loud. Catalog data is from [TMDB](https://www.themoviedb.org/). This product uses the TMDB API but is not endorsed or certified by TMDB. Official logo: `assets/tmdb_square.svg`. She does **not** scrape Netflix or Amazon. Put `TMDB_API_KEY` and `TMDB_READ_TOKEN` in local `config.json` only.

## Secrets

API keys stay in local `config.json` only. Owner identity (mail, GitHub, standing Bot briefs) stays in local `owner.json`. This repo ships empty placeholders in `config.example.json` and `owner.example.json`. Never copy `config.json`, `license.json`, or `owner.json` into a store zip.
