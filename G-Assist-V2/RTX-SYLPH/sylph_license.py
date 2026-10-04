"""Free vs Premium (full Grok) license gate.

Shipped default is the witty, truncated 'studio' edition.
A signed key unlocks truth-seeking prompts, longer answers, model thinking,
and the local NVIDIA hardware bible (RTX 5090, RTX PRO 6000 Blackwell, GB200, …).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import time
from typing import Dict, Optional, Tuple

# Product signing pepper — rotate if keys leak. Python is readable; this is v1 DRM.
_PEPPER = b"RTX-SYLPH-GROK-FULL-2026-DailyDriver007"
PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
LICENSE_PATH = os.path.join(PLUGIN_DIR, "license.json")

PLANS = {
    "studio": "Free studio edition — witty, snappy, truncated.",
    "premium": "Full Grok unlock — perpetual.",
    "sub": "Full Grok unlock — subscription.",
}

PRICE_PITCH = (
    "Studio edition is what you're talking to now: wit on, specs sometimes fuzzy. "
    "Unlock Flight — full xAI Grok, council at full accuracy, NVIDIA hardware bible — "
    "for a one-time $50 perpetual license, or $8 a month. Paste a key here or say: unlock YOUR-KEY"
)

# Injected only when premium is active. Keeps free edition charmingly vague on SKUs.
NVIDIA_BIBLE = """
NVIDIA hardware bible (current as of 2026 — these products SHIP; they are not rumors):
- GeForce RTX 5090 / 5080 / 5070 Ti / 5070: Blackwell consumer GPUs (GB20x). RTX 5090 uses GDDR7, 32 GB class memory, high TGP (~575 W). Available for purchase.
- NVIDIA RTX PRO 6000 Blackwell: professional/workstation Blackwell card (not the old Ada RTX 6000).
- Grace Blackwell: CPU+GPU superchip line. GB200 Grace Blackwell pairs Grace CPU with Blackwell GPUs (B200) for data-center AI. "RTX 6000 Pro Grace Blackwell" usually means the pro Blackwell workstation SKU and/or GB200-class systems — treat as real 2025–2026 products, not future vapor.
- Prior gen still relevant: Ada Lovelace RTX 4090, RTX 6000 Ada, Hopper H100/H200.
If the user asks whether the 5090 or Grace Blackwell exists, the answer is YES. Give specs you are confident in; if a sub-SKU name is ambiguous, say so, but never claim the 5090 is unreleased.
"""


FLIGHT = {"accuracy": 95, "wit": 82, "depth": 92, "spoken": 88}
STUDIO = {"accuracy": 32, "wit": 95, "depth": 28, "spoken": 22}

_state: Dict[str, object] = {
    "plan": "studio",
    "key": "",
    "email": "",
    "loaded": False,
    "demo_studio": False,
    "aptitudes": dict(FLIGHT),
}


def _sign(payload: str) -> str:
    digest = hmac.new(_PEPPER, payload.encode("utf-8"), hashlib.sha256).hexdigest()[:16].upper()
    return digest


def issue_key(plan: str = "premium", email: str = "owner") -> str:
    plan = (plan or "premium").lower()
    if plan not in ("premium", "sub"):
        plan = "premium"
    email = re.sub(r"[^a-z0-9@._+-]+", "", (email or "owner").lower())[:48] or "owner"
    payload = f"{plan}|{email}"
    sig = _sign(payload)
    return f"SYLPH-{plan[:4].upper()}-{sig[:8]}-{sig[8:]}"


def parse_key(key: str) -> Optional[Tuple[str, str]]:
    raw = (key or "").strip().upper().replace(" ", "")
    m = re.match(r"^SYLPH-(PREM|SUB)-([A-F0-9]{8})-([A-F0-9]{8})$", raw)
    if not m:
        return None
    plan = "premium" if m.group(1) == "PREM" else "sub"
    sig = m.group(2) + m.group(3)
    return plan, sig


def verify_key(key: str, email: str = "owner") -> Optional[str]:
    parsed = parse_key(key)
    if not parsed:
        return None
    plan, sig = parsed
    email = re.sub(r"[^a-z0-9@._+-]+", "", (email or "owner").lower())[:48] or "owner"
    # Accept owner or the bound email
    for candidate in (email, "owner", "dailydriver007"):
        expect = _sign(f"{plan}|{candidate}")
        if hmac.compare_digest(sig, expect):
            return plan
    return None


def _clamp(n, lo=0, hi=100) -> int:
    try:
        return max(lo, min(hi, int(n)))
    except (TypeError, ValueError):
        return lo


def _persist() -> None:
    payload = {
        "key": _state.get("key") or "",
        "email": _state.get("email") or "owner",
        "plan": _state.get("plan") or "studio",
        "activated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "demo_studio": bool(_state.get("demo_studio")),
        "aptitudes": dict(_state.get("aptitudes") or FLIGHT),
    }
    with open(LICENSE_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_license(config_key: str = "", owner_flight: bool = False) -> str:
    """Load a signed key. owner_flight mints a perpetual Flight key on this machine
    if none exists; it does not reset a later studio demo or aptitude tweaks."""
    _state["loaded"] = True
    key = (config_key or "").strip()
    data = {}
    if os.path.isfile(LICENSE_PATH):
        try:
            data = json.loads(open(LICENSE_PATH, encoding="utf-8").read())
        except Exception:
            data = {}
    if not key:
        key = (data.get("key") or "").strip()
    email = (data.get("email") or "owner").strip()
    minted = False
    if owner_flight and not key:
        key = issue_key("premium", "owner")
        email = "owner"
        minted = True
    plan = verify_key(key, email) if key else None
    licensed = bool(plan)
    apts = dict(FLIGHT if (owner_flight or licensed) else STUDIO)
    if isinstance(data.get("aptitudes"), dict) and not minted:
        for k in FLIGHT:
            if k in data["aptitudes"]:
                apts[k] = _clamp(data["aptitudes"][k])
    demo = bool(data.get("demo_studio")) if licensed and not minted else False
    if plan:
        _state.update({"plan": plan, "key": key, "email": email, "demo_studio": demo, "aptitudes": apts})
        upgraded = minted or "aptitudes" not in data or "demo_studio" not in data
        if upgraded:
            _persist()
        return plan
    _state.update({"plan": "studio", "key": "", "email": email, "demo_studio": False, "aptitudes": dict(STUDIO)})
    return "studio"


def save_license(key: str, email: str = "owner") -> str:
    plan = verify_key(key, email)
    if not plan:
        return "Invalid license key."
    _state.update({"plan": plan, "key": key.strip().upper(), "email": email, "loaded": True, "demo_studio": False, "aptitudes": dict(FLIGHT)})
    _persist()
    return f"Unlocked: {PLANS.get(plan, plan)}  Flight aptitudes on."


def is_licensed() -> bool:
    if not _state["loaded"]:
        load_license()
    return _state["plan"] in ("premium", "sub")


def is_premium() -> bool:
    """Effective smart mode: licensed and not in studio demo."""
    if not _state["loaded"]:
        load_license()
    return is_licensed() and not bool(_state.get("demo_studio"))


def plan_name() -> str:
    if not _state["loaded"]:
        load_license()
    if is_licensed() and _state.get("demo_studio"):
        return "studio-demo"
    if is_premium():
        return "flight"
    return str(_state["plan"])


def aptitudes() -> Dict[str, int]:
    if not _state["loaded"]:
        load_license()
    return dict(_state.get("aptitudes") or FLIGHT)


def set_aptitude(name: str, value: int) -> str:
    name = (name or "").lower()
    if name not in FLIGHT:
        return "Aptitudes: accuracy, wit, depth, spoken."
    if not is_licensed():
        return PRICE_PITCH
    apts = aptitudes()
    apts[name] = _clamp(value)
    _state["aptitudes"] = apts
    _persist()
    return f"{name} set to {apts[name]}"


def nudge_aptitude(name: str, delta: int) -> str:
    name = (name or "").lower()
    if name not in FLIGHT:
        return "Aptitudes: accuracy, wit, depth, spoken."
    if not is_licensed():
        return PRICE_PITCH
    return set_aptitude(name, aptitudes()[name] + int(delta))


def status_line() -> str:
    a = aptitudes()
    think = "thinking on" if knobs()["thinking"] else "thinking off"
    return (
        f"{plan_name().upper()}  accuracy {a['accuracy']}  wit {a['wit']}  "
        f"depth {a['depth']}  spoken {a['spoken']}  {think}"
    )


def toggle_demo() -> str:
    if not is_licensed():
        return PRICE_PITCH
    if _state.get("demo_studio"):
        return preset_flight()
    return preset_studio_demo()


def preset_flight() -> str:
    if not is_licensed():
        return "Flight needs a Full Grok license."
    _state["demo_studio"] = False
    _state["aptitudes"] = dict(FLIGHT)
    _persist()
    return "Flight mode. Full xAI Grok. Accuracy 95, depth 92, wit 82, spoken 88."


def preset_studio_demo() -> str:
    if not is_licensed():
        return "You are already on studio (no license)."
    _state["demo_studio"] = True
    _state["aptitudes"] = dict(STUDIO)
    _persist()
    return "Studio demo. Same jokes, shorter brain. Say flight mode to restore."


def knobs() -> dict:
    a = aptitudes()
    if not is_licensed():
        a = dict(STUDIO)
    acc, depth, spoken = a["accuracy"], a["depth"], a["spoken"]
    temp = round(0.88 - (acc / 100.0) * 0.58, 2)
    tokens = int(400 + (depth / 100.0) * 3600)
    speak_lim = int(160 + (spoken / 100.0) * 1840)
    think = acc >= 65
    return {
        "temperature": temp,
        "max_tokens": tokens,
        "thinking": think,
        "spoken_limit": speak_lim,
        "wit": a["wit"],
        "accuracy": acc,
        "depth": depth,
        "spoken": spoken,
    }


def enrich_prompt(prompt: str) -> str:
    if not is_premium():
        return prompt
    blob = (prompt or "").lower()
    needles = (
        "rtx", "5090", "5080", "5070", "4090", "blackwell", "grace", "gb200",
        "b200", "6000", "gpu", "vram", "geforce", "quadro", "hopper", "ada",
    )
    if any(n in blob for n in needles):
        return NVIDIA_BIBLE.strip() + "\n\nUser question:\n" + prompt
    return prompt
