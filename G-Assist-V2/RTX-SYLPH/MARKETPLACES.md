# RTX SYLPH — packaging, marketplaces, mobile

## What you can sell

Sell the **companion** (HUD, voice, desk, animations, installer), not other companies’ APIs.

Never ship `config.json` with keys. Buyer pastes **their** Grok / Gemini / NVIDIA / TMDB / HA keys.

Clips stay a Drive pack or a paid **asset DLC** (your Imagine likeness). Plugin code can be cheaper or free; the holographic pack is the premium SKU.

## Package (v1 store zip)

Inno Setup or NSIS installer that:

1. Bundles embeddable Python 3.10 + `requirements.txt` wheels + QuadCast PyAudio wheel  
2. Copies plugin to `%PROGRAMDATA%\NVIDIA Corporation\nvtopps\rise\plugins\RTX-SYLPH` (needs admin)  
3. Writes `config.example.json` → `config.json` and opens a key wizard  
4. Fetches or points at the Drive animation pack  
5. Drops `RTX SYLPH.lnk` on the desktop  
6. Excludes secrets, `council_windows/`, `__pycache__/`, `*.log`, raw `.mp4` if hosted separately  

**Never ship (owner machine only):** `config.json`, `license.json`, `owner.json`, `desk_layout.json`, `sylph_crash.log`. Those hold API keys, the perpetual Flight unlock, mail/GitHub identity, and local layout. The zip uses `config.example.json` (`SYLPH_OWNER_FLIGHT: false`) and `owner.example.json` (empty). Buyer pastes **their** keys and Bot connectors. Your Gmail, GitHub, Stripe, Minewing briefs, and owner Flight flag stay on this PC.  

Standalone SKU (no G-Assist): same payload, launch `run_sylph.bat` only.

## Where to list

| Place | Fit | Notes |
|---|---|---|
| **NVIDIA G-Assist on mod.io** | Best first | She is a G-Assist V2 plugin. Often **free or donation**. Follow NVIDIA plugin rules. |
| **Gumroad / Itch.io** | Best paid | Standalone HUD + clip pack. Simple seller TOS. You keep the customer list. |
| **Ko-fi / Patreon** | Recurring | Early clips, voice presets, new seats. |
| **Steam** | Later | Direct-to-desktop companion. Review, trailer, 18+ if needed. Heavy for v1. |
| **Microsoft Store** | Hard | MSIX, certification, sandbox vs pygame/mic/admin ProgramData. |
| **Overwolf / Nexus** | Weak | Game overlays, not her. |
| **Chrome Web Store** | No | She is not an extension. |

Lead with **mod.io (G-Assist) + Gumroad (paid standalone + assets)**.

Price ballpark: free plugin on mod.io; **$12–25** standalone studio; **$8–15** clip-pack DLC if the plugin is free.

**Flight / Full Grok unlock (the real SKU):** studio ships witty but truncated (the “carbon fiber / light as a lie” rehearsal). A license key switches her to **Flight** — full xAI Grok, truth-seeking prompts, longer answers, model thinking, and a local NVIDIA hardware bible (RTX 5090, RTX PRO 6000 Blackwell, Grace Blackwell / GB200). Licensed buyers can still **demo studio** and slide aptitudes (accuracy, wit, depth, spoken) on the PRO tile. **$50 perpetual** or **$8/month**. Keys are `SYLPH-PREM-…` / `SYLPH-SUB-…` issued with `issue_license.py`. Gumroad can email the key after payment. Shipped `config.example.json` keeps `SYLPH_OWNER_FLIGHT` false so store copies stay studio until the buyer pastes a key.

**Grok Bot (SuperGrok Plus):** SYLPH is the desktop hologram (wake word, GPU, house, council). Grok Bot is the xAI teammate with its own cloud computer that signs into tools. SuperGrok Plus includes Bot; as of Aug 2026 there is **no public Bot API**. The desk **BOT** tile / `open grok bot` launches `https://x.ai/bot`. When xAI ships webhooks or MCP, that tile becomes a handoff: SYLPH stays on the pad, Bot flies the job.

## Legal on listings

- Not affiliated with NVIDIA, Google, xAI, Netflix, Amazon, or TMDB beyond their public APIs and required TMDB attribution.  
- Buyer uses their own streaming accounts and API keys.  
- Your likeness/clips: you license, they don’t resell.

## Mobile version

**Not the current pygame HUD.** OpenCV, PyAudio, Win32 hotkeys, Chrome council windows, and a frameless desktop overlay do not ship as an iOS/Android app.

Realistic mobile SKUs:

1. **Companion remote (best v1 mobile)** — Flutter or React Native: talk/text to a SYLPH instance on the PC (local LAN WebSocket). Phone is the mic + tiles; the hologram stays on the desktop.  
2. **PWA** — console + watch + weather + calendar in a browser. No wake-word hologram.  
3. **Full mobile hologram** — Unreal/Unity or Filament + on-device or cloud LLM. New product, not a port.

Do **not** promise App Store “RTX SYLPH” as a 1:1 port of this repo.

LAN remote is the honest next mobile step: same brain, phone as a controller.
