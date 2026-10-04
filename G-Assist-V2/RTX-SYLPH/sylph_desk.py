"""SYLPH utility dock — compact tiles that expand in place."""
from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from threading import Thread
from typing import Callable, Deque, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger("rtx_sylph")

quiet_hours = False
focus_until = 0.0
focus_minutes = 25
expanded = "gpu"
home_env = "ha"
gpu_hist: Deque[Tuple[float, float]] = deque(maxlen=48)
gpu_occupancy_snap: Dict[str, object] = {}
clipboard: List[str] = []
links: List[Dict[str, str]] = []
weather: Dict[str, object] = {}
events: List[Dict[str, str]] = []
media: List[str] = []
ha_tiles: List[Dict[str, str]] = []
_last = {"wx": 0.0, "cal": 0.0, "gpu": 0.0, "occ": 0.0, "clip": 0.0, "media": 0.0, "ha": 0.0}
_host = None
_paint: Optional[Callable] = None
_dirty = False
_dock = None
_reset_dock: Optional[Callable] = None

TILES = [
    ("gpu", "GPU"),
    ("wx", "WX"),
    ("cal", "CAL"),
    ("focus", "FOCUS"),
    ("media", "MEDIA"),
    ("clip", "CLIP"),
    ("home", "HOME"),
    ("links", "LINKS"),
    ("quiet", "QUIET"),
    ("shot", "SHOT"),
    ("bot", "BOT"),
    ("mail", "GMAIL"),
    ("gcal", "GCAL"),
    ("drive", "DRV"),
    ("github", "GH"),
    ("stripe", "PAY"),
    ("unlock", "PRO"),
]

# Grok Bot connectors (OAuth lives in the Bot app). SYLPH only hands a brief + opens Bot.
BOT_CREW = (
    ("mail", "GMAIL"),
    ("cal", "GCAL"),
    ("drive", "DRIVE"),
    ("github", "GH"),
    ("stripe", "PAY"),
)

_OWNER: Dict[str, str] = {}


def _owner() -> Dict[str, str]:
    global _OWNER
    if _OWNER:
        return _OWNER
    path = os.path.join(_plugin_dir(), "owner.json")
    data: Dict[str, str] = {}
    if os.path.isfile(path):
        try:
            loaded = json.loads(open(path, encoding="utf-8").read())
            if isinstance(loaded, dict):
                data = {str(k): str(v) if v is not None else "" for k, v in loaded.items()}
        except Exception:
            data = {}
    _OWNER = data
    return _OWNER


def _owner_name() -> str:
    return (_owner().get("display_name") or "the owner").strip() or "the owner"


def _mail_account() -> str:
    return (_owner().get("mail_account") or "").strip()


def _mail_standing() -> str:
    custom = (_owner().get("mail_standing") or "").strip()
    if custom:
        return custom
    acct = _mail_account() or "the owner's Gmail"
    name = _owner_name()
    return (
        f"You are MAIL, {name}'s Grok Bot.\n"
        f"Inbox: {acct} only. Do not open other accounts.\n\n"
        "READ: Summarize unread / recent mail out loud-ready: from, subject, date, 2-line gist.\n"
        "Never read API keys, passwords, license keys, or .env out loud. Say \"secret, skipped.\"\n\n"
        "DRAFT: Put replies in Gmail DRAFTS. NEVER send. NEVER reply-all unless they said so.\n"
        "NEVER attach source, keys, or secrets.\n"
        "If extra instructions are below, follow those for this turn.\n"
    )


def set_clipboard(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False
    try:
        import win32clipboard
        import win32con

        win32clipboard.OpenClipboard()
        try:
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(win32con.CF_UNICODETEXT, raw)
        finally:
            win32clipboard.CloseClipboard()
        clipboard.insert(0, raw)
        del clipboard[20:]
        mark()
        return True
    except Exception:
        logger.warning("clipboard set failed", exc_info=True)
        return False


def mail_brief(extra: str = "") -> str:
    extra = (extra or "").strip()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    default_turn = (_owner().get("mail_default_turn") or "unread. Draft, do not send.").strip()
    tail = f"\nThis turn: {extra}\n" if extra else f"\nThis turn: {default_turn}\n"
    return _mail_standing() + f"\nLocal time: {now}\n" + tail


def bot_brief(job: str, extra: str = "") -> str:
    job = (job or "").strip().lower()
    extra = (extra or "").strip()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    if job == "mail":
        return mail_brief(extra)
    if job == "watch":
        body = (
            "You are Bot WATCH for RTX SYLPH's owner.\n"
            "Catalog is TMDB public watch-provider data. Do not scrape Netflix or Amazon.\n"
            "Suggest what to watch. Short list, full synopsis if asked. Attribute TMDB.\n"
        )
    elif job in ("cal", "gcal", "calendar"):
        body = (
            "You are Bot CAL / Google Calendar for RTX SYLPH's owner.\n"
            "Use the Google Calendar connector already added in Grok Bot.\n"
            "Search events, propose times, draft (do not send) meeting invites.\n"
            "Do not email attendees unless he said so.\n"
        )
        try:
            calendar_refresh(True)
            body += "SYLPH ICS snapshot: " + calendar_line() + "\n"
        except Exception:
            pass
    elif job in ("drive", "gdrive"):
        body = (
            "You are Bot DRIVE for RTX SYLPH's owner.\n"
            "Use the Google Drive connector already added in Grok Bot.\n"
            "Find, summarize, and organize files. Do not delete or share outside the org unless he said so.\n"
            "Never paste API keys, license.json, or .env into chat.\n"
        )
    elif job in ("github", "gh"):
        who = _owner_name()
        gh = (_owner().get("github") or "").strip()
        org = (_owner().get("org") or "").strip()
        tag = f" ({gh}" + (f" / {org}" if org else "") + ")" if gh else ""
        body = (
            f"You are Bot GH for {who}{tag}.\n"
            "Use the GitHub connector in Grok Bot. If it still wants a token: GitHub → Settings → "
            "Developer settings → Personal access tokens → classic, repo + read:org. "
            "Put the token in Grok Bot, never in SYLPH config or chat.\n"
            "Issues, PRs, repo status. Do not force-push. Do not leak secrets.\n"
        )
    elif job in ("stripe", "pay"):
        body = (
            "You are Bot PAY / Stripe for RTX SYLPH's owner.\n"
            "Use the Stripe connector already added in Grok Bot.\n"
            "Read products, prices, customers, and test vs live. "
            "Do not capture charges or change live prices unless he said so in this turn.\n"
            "RTX SYLPH Flight is $50 perpetual / $8 month — treat that as the product SKU if asked.\n"
        )
    elif job in ("home", "ha"):
        body = (
            "You are Bot HOME for RTX SYLPH's owner.\n"
            "Lights, climate, locks — propose the spoken command for SYLPH. Do not claim you flipped a switch.\n"
        )
    elif job in ("wx", "weather"):
        body = (
            "You are Bot WX for RTX SYLPH's owner.\n"
            "Weather and AQI. Be specific. Suggest indoor vs outdoor.\n"
        )
        try:
            weather_refresh(True)
            body += "Now: " + weather_line() + "\n"
        except Exception:
            pass
    else:
        body = f"You are a Grok Bot attached to RTX SYLPH ({job or 'job'}).\n"
    if extra:
        body += f"This turn: {extra}\n"
    return body + f"Local time: {now}\n"


def handoff_bot(job: str, extra: str = "") -> str:
    brief = bot_brief(job, extra)
    set_clipboard(brief)
    mark()
    return f"GROK_JOB {job}"


def attach(host) -> None:
    global _host, home_env, links
    _host = host
    home_env = (getattr(host, "SMART_HOME_ENV", "ha") or "ha").lower()
    links[:] = _load_links()


def mark() -> None:
    global _dirty
    _dirty = True


def is_quiet() -> bool:
    return quiet_hours


def is_quiet_wake(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ("i'm back", "im back", "cancel quiet", "wake up", "end quiet", "stop quiet"))


def _plugin_dir() -> str:
    return getattr(_host, "_plugin_dir", os.getcwd()) if _host else os.getcwd()


def _bot_url() -> str:
    url = ""
    if _host:
        url = str(getattr(_host, "GROK_BOT_URL", "") or "")
    return url.strip() or "https://x.ai/bot"


def _layout_path() -> str:
    return os.path.join(_plugin_dir(), "desk_layout.json")


def _clamp_xy(x: int, y: int, w: int, h: int, sw: int, sh: int) -> Tuple[int, int]:
    x, y = int(x), int(y)
    if sw > w > 0:
        x = max(0, min(x, sw - w))
    else:
        x = max(0, x)
    if sh > h > 0:
        y = max(0, min(y, sh - h))
    else:
        y = max(0, y)
    return x, y


def _load_layout(sw: int, sh: int, w: int, h: int) -> Tuple[int, int]:
    default = (max(8, (sw - w) // 2), 8)
    path = _layout_path()
    if not os.path.isfile(path):
        return default
    try:
        data = json.loads(open(path, encoding="utf-8").read())
        return _clamp_xy(int(data.get("x", default[0])), int(data.get("y", default[1])), w, h, sw, sh)
    except Exception:
        return default


def _save_layout(x: int, y: int) -> None:
    try:
        with open(_layout_path(), "w", encoding="utf-8") as handle:
            json.dump({"x": int(x), "y": int(y)}, handle)
    except Exception:
        logger.debug("desk layout save failed", exc_info=True)


def _links_path() -> str:
    folder = os.path.join(_plugin_dir(), "council_windows")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, "links.json")


def _load_links() -> List[Dict[str, str]]:
    path = _links_path()
    try:
        if os.path.isfile(path):
            data = json.loads(open(path, encoding="utf-8").read())
            if isinstance(data, list):
                return data[-40:]
    except Exception:
        pass
    return []


def save_link(url: str, title: str = "") -> None:
    url = (url or "").strip()
    if not url.startswith("http"):
        return
    item = {"url": url, "title": title or url, "ts": time.strftime("%Y-%m-%d %H:%M")}
    links.append(item)
    try:
        open(_links_path(), "w", encoding="utf-8").write(json.dumps(links[-40:], indent=2))
    except Exception as e:
        logger.warning("Link shelf save failed: %s", e)
    mark()


def sample_gpu() -> None:
    global gpu_occupancy_snap
    try:
        import GPUtil

        gpus = GPUtil.getGPUs()
        if not gpus:
            return
        g = gpus[0]
        gpu_hist.append((g.load * 100.0, float(g.temperature or 0)))
    except Exception:
        pass
    now = time.time()
    if now - _last.get("occ", 0) >= 8.0:
        _last["occ"] = now
        try:
            vg = os.environ.get("MACH01_ROOT") or os.environ.get("VOICE_GROK_ROOT") or ""
            import sys

            if not vg or not os.path.isdir(vg):
                raise RuntimeError("set MACH01_ROOT to enable the occupancy module")
            if vg not in sys.path:
                sys.path.insert(0, vg)
            from gpu_occupancy import snapshot, write_bus

            gpu_occupancy_snap = snapshot()
            write_bus(gpu_occupancy_snap)
        except Exception:
            pass


def weather_refresh(force: bool = False) -> None:
    now = time.time()
    if not force and now - _last["wx"] < 600:
        return
    _last["wx"] = now
    lat = getattr(_host, "WEATHER_LAT", "") if _host else ""
    lon = getattr(_host, "WEATHER_LON", "") if _host else ""
    try:
        if not lat or not lon:
            geo = requests.get("http://ip-api.com/json/?fields=lat,lon,city,status", timeout=6).json()
            if geo.get("status") == "success":
                lat, lon = geo.get("lat"), geo.get("lon")
                weather["city"] = geo.get("city") or ""
        if lat in (None, "") or lon in (None, ""):
            weather["err"] = "set WEATHER_LAT / WEATHER_LON"
            mark()
            return
        wx = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,weather_code,wind_speed_10m,relative_humidity_2m",
                "temperature_unit": "fahrenheit",
            },
            timeout=8,
        ).json()
        aq = requests.get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={"latitude": lat, "longitude": lon, "current": "us_aqi,pm2_5"},
            timeout=8,
        ).json()
        cur = wx.get("current") or {}
        aqc = aq.get("current") or {}
        weather.update(
            {
                "temp": cur.get("temperature_2m"),
                "code": cur.get("weather_code"),
                "wind": cur.get("wind_speed_10m"),
                "rh": cur.get("relative_humidity_2m"),
                "aqi": aqc.get("us_aqi"),
                "pm25": aqc.get("pm2_5"),
                "err": "",
            }
        )
        mark()
    except Exception as e:
        weather["err"] = str(e)
        mark()


_WX = {
    0: "clear",
    1: "mostly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    51: "drizzle",
    61: "rain",
    71: "snow",
    80: "showers",
    95: "thunder",
}


def weather_line() -> str:
    if weather.get("err"):
        return f"Weather: {weather['err']}"
    city = weather.get("city") or "here"
    temp = weather.get("temp")
    aqi = weather.get("aqi")
    code = _WX.get(int(weather.get("code") or 0), "mixed")
    if temp is None:
        return "Weather not ready."
    aqi_s = f", AQI {int(aqi)}" if aqi is not None else ""
    return f"{city}: {temp:.0f}°F, {code}{aqi_s}."


def _parse_ics(raw: str) -> List[Dict[str, str]]:
    text = raw.replace("\r\n ", "").replace("\n ", "")
    out = []
    for block in text.split("BEGIN:VEVENT")[1:]:
        def field(name: str) -> str:
            m = re.search(rf"^{name}[^:]*:(.+)$", block, re.M)
            return (m.group(1).strip() if m else "")

        start = field("DTSTART")
        summary = field("SUMMARY")
        if not summary:
            continue
        out.append({"start": start, "summary": summary})
    return out[:12]


def _ics_when(stamp: str) -> str:
    digits = re.sub(r"[^0-9]", "", stamp or "")[:14]
    if len(digits) < 8:
        return stamp
    try:
        if "T" in stamp or len(digits) >= 14:
            dt = datetime.strptime(digits[:14], "%Y%m%d%H%M%S")
            if stamp.endswith("Z"):
                dt = dt.replace(tzinfo=timezone.utc).astimezone()
            return dt.strftime("%a %H:%M")
        dt = datetime.strptime(digits[:8], "%Y%m%d")
        return dt.strftime("%a all-day")
    except Exception:
        return stamp[:16]


def calendar_refresh(force: bool = False) -> None:
    now = time.time()
    if not force and now - _last["cal"] < 300:
        return
    _last["cal"] = now
    url = getattr(_host, "CALENDAR_ICS_URL", "") if _host else ""
    if not url:
        events[:] = []
        mark()
        return
    try:
        resp = requests.get(url, timeout=10)
        events[:] = _parse_ics(resp.text if resp.status_code < 400 else "")
        mark()
    except Exception as e:
        events[:] = [{"start": "", "summary": f"Calendar error: {e}"}]
        mark()


def calendar_line() -> str:
    if not events:
        return "Calendar: add CALENDAR_ICS_URL (Google Calendar secret iCal address) to config.json."
    top = events[:3]
    return "Next: " + "; ".join(f"{_ics_when(e['start'])} {e['summary']}" for e in top)


def clip_poll() -> None:
    now = time.time()
    if now - _last["clip"] < 1.2:
        return
    _last["clip"] = now
    try:
        import win32clipboard
        import win32con

        win32clipboard.OpenClipboard()
        try:
            if win32clipboard.IsClipboardFormatAvailable(win32con.CF_UNICODETEXT):
                text = win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
            else:
                text = ""
        finally:
            win32clipboard.CloseClipboard()
        text = (text or "").strip()
        if text and (not clipboard or clipboard[0] != text) and len(text) < 4000:
            clipboard.insert(0, text)
            del clipboard[20:]
            mark()
    except Exception:
        pass


def media_poll() -> None:
    now = time.time()
    if now - _last["media"] < 2.5:
        return
    _last["media"] = now
    rows = []
    try:
        from pycaw.pycaw import AudioUtilities

        for session in AudioUtilities.GetAllSessions():
            proc = session.Process
            if not proc:
                continue
            vol = session.SimpleAudioVolume
            level = int((vol.GetMasterVolume() or 0) * 100)
            mute = bool(vol.GetMute())
            name = proc.name().replace(".exe", "")
            if level <= 0 and not mute:
                continue
            rows.append(f"{name} {level}%{' mute' if mute else ''}")
    except Exception:
        rows = ["Install pycaw for per-app mixer (py -3.10 -m pip install pycaw)."]
    media[:] = rows[:8]
    mark()


def ha_poll(force: bool = False) -> None:
    now = time.time()
    if not force and now - _last["ha"] < 8:
        return
    _last["ha"] = now
    url = getattr(_host, "HA_URL", "") if _host else ""
    key = getattr(_host, "HA_KEY", "") if _host else ""
    if not url or not key:
        ha_tiles[:] = []
        mark()
        return
    try:
        resp = requests.get(
            f"{url.rstrip('/')}/api/states",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=8,
        )
        if resp.status_code >= 400:
            ha_tiles[:] = [{"id": "err", "name": f"HA HTTP {resp.status_code}", "state": ""}]
            mark()
            return
        wanted = []
        for st in resp.json():
            eid = st.get("entity_id") or ""
            if eid.startswith(("light.", "switch.", "climate.", "fan.", "lock.")):
                name = (st.get("attributes") or {}).get("friendly_name") or eid.split(".", 1)[-1]
                wanted.append({"id": eid, "name": name, "state": str(st.get("state") or "")})
            if len(wanted) >= 8:
                break
        ha_tiles[:] = wanted
        mark()
    except Exception as e:
        ha_tiles[:] = [{"id": "err", "name": str(e), "state": ""}]
        mark()


def ha_toggle(entity_id: str) -> str:
    url = getattr(_host, "HA_URL", "") if _host else ""
    key = getattr(_host, "HA_KEY", "") if _host else ""
    if not url or not key or "." not in entity_id:
        return "Home Assistant not configured"
    domain = entity_id.split(".", 1)[0]
    service = "toggle" if domain in ("light", "switch", "fan") else "turn_on"
    try:
        requests.post(
            f"{url.rstrip('/')}/api/services/{domain}/{service}",
            json={"entity_id": entity_id},
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            timeout=8,
        )
        ha_poll(force=True)
        return f"Toggled {entity_id}"
    except Exception as e:
        return str(e)


def start_focus(minutes: int = 25) -> str:
    global focus_until, focus_minutes
    focus_minutes = max(1, min(180, int(minutes)))
    focus_until = time.time() + focus_minutes * 60
    mark()
    return f"Focus {focus_minutes} minutes. I'll ping you."


def cancel_focus() -> str:
    global focus_until
    focus_until = 0.0
    mark()
    return "Focus cancelled."


def set_quiet(on: bool) -> str:
    global quiet_hours
    quiet_hours = bool(on)
    mark()
    if quiet_hours:
        return "Quiet hours on. Say sylph I'm back when you want me listening."
    return "Quiet hours off. I'm listening."


def focus_tick() -> None:
    global focus_until
    if focus_until and time.time() >= focus_until:
        focus_until = 0.0
        mark()
        if _host:
            _host.speak("Focus session done. Stretch.")
            _host.chat_note("SYS", "Focus session done.")


def on_tick() -> None:
    now = time.time()
    if now - _last["gpu"] >= 1.0:
        _last["gpu"] = now
        sample_gpu()
        mark()
    focus_tick()
    if expanded == "wx":
        weather_refresh()
    if expanded == "cal":
        calendar_refresh()
    if expanded == "clip":
        clip_poll()
    if expanded == "media":
        media_poll()
    if expanded == "home" and home_env == "ha":
        ha_poll()
    global _dirty
    if _dirty and _paint:
        _dirty = False
        try:
            _paint()
        except Exception:
            pass


def _bg() -> str:
    return "#001114"


def build_dock(tk, parent, sw: int, sh: int, host) -> object:
    global _paint, expanded, _dock, _reset_dock
    attach(host)
    dock = tk.Toplevel(parent)
    dock.overrideredirect(True)
    dock.attributes("-topmost", True)
    dock.configure(bg=_bg())
    width, collapsed, expanded_h, pro_h = 1240, 54, 268, 348
    pos = {"x": 0, "y": 0}
    pos["x"], pos["y"] = _load_layout(sw, sh, width, collapsed)
    dock.geometry(f"{width}x{collapsed}+{pos['x']}+{pos['y']}")
    _dock = dock

    bar = tk.Frame(dock, bg=_bg(), cursor="fleur")
    bar.pack(fill="x", padx=6, pady=4)
    title = tk.Label(
        bar,
        text="☰  SYLPH  DESK",
        fg="#00ffd2",
        bg=_bg(),
        font=("Consolas", 9, "bold"),
        cursor="fleur",
    )
    title.pack(side="left", padx=(4, 8))

    body = tk.Text(
        dock,
        bg="#00181c",
        fg="#d8fff4",
        font=("Consolas", 9),
        wrap="word",
        relief="flat",
        height=10,
        padx=8,
        pady=6,
    )
    body.pack(fill="both", expand=True, padx=8, pady=(0, 6))
    body.tag_config("H", foreground="#ffe066", font=("Consolas", 9, "bold"))
    body.tag_config("F", foreground="#66aa99")
    body.pack_forget()

    btns: Dict[str, object] = {}
    pro_scales: Dict[str, object] = {}
    apt_vars: Dict[str, object] = {}
    header_var = tk.StringVar(value="")
    _pro_ready = {"on": False}
    _pro_armed = {"on": False}

    pro_frame = tk.Frame(dock, bg="#00181c")
    tk.Label(
        pro_frame,
        text="FLIGHT  /  STUDIO  DEMO",
        fg="#ffe066",
        bg="#00181c",
        font=("Consolas", 9, "bold"),
        anchor="w",
    ).pack(fill="x", padx=4, pady=(2, 0))
    tk.Label(
        pro_frame,
        textvariable=header_var,
        fg="#d8fff4",
        bg="#00181c",
        font=("Consolas", 8),
        wraplength=880,
        justify="left",
        anchor="w",
    ).pack(fill="x", padx=4, pady=(0, 4))
    pro_btns = tk.Frame(pro_frame, bg="#00181c")
    pro_btns.pack(fill="x", padx=4, pady=(0, 4))

    def _do_flight():
        import sylph_license

        msg = sylph_license.preset_flight()
        refresh_pro(force_scales=True)
        if _host:
            _host.speak(msg)
            _host.chat_note("SYS", msg)

    def _do_demo():
        import sylph_license

        msg = sylph_license.preset_studio_demo()
        refresh_pro(force_scales=True)
        if _host:
            _host.speak(msg)
            _host.chat_note("SYS", msg)

    tk.Button(
        pro_btns,
        text="FLIGHT",
        command=_do_flight,
        bg="#003a3a",
        fg="#00ffd2",
        activebackground="#00ffd2",
        activeforeground="#001114",
        relief="flat",
        font=("Consolas", 8, "bold"),
        padx=10,
        pady=2,
    ).pack(side="left", padx=(0, 6))
    tk.Button(
        pro_btns,
        text="DEMO STUDIO",
        command=_do_demo,
        bg="#3a2800",
        fg="#ffd27a",
        activebackground="#ffd27a",
        activeforeground="#001114",
        relief="flat",
        font=("Consolas", 8, "bold"),
        padx=10,
        pady=2,
    ).pack(side="left")

    def _on_apt(name: str):
        def _cb(val):
            import sylph_license

            if not _pro_armed["on"] or not sylph_license.is_licensed():
                return
            sylph_license.set_aptitude(name, int(float(val)))
            header_var.set(sylph_license.status_line())
            _sync_pro_tile()

        return _cb

    for apt_name in ("accuracy", "wit", "depth", "spoken"):
        row = tk.Frame(pro_frame, bg="#00181c")
        row.pack(fill="x", padx=4, pady=1)
        tk.Label(
            row,
            text=apt_name.upper(),
            width=10,
            anchor="w",
            fg="#00ffd2",
            bg="#00181c",
            font=("Consolas", 8, "bold"),
        ).pack(side="left")
        var = tk.IntVar(value=0)
        scale = tk.Scale(
            row,
            from_=0,
            to=100,
            orient="horizontal",
            variable=var,
            command=_on_apt(apt_name),
            length=620,
            showvalue=True,
            bg="#00181c",
            fg="#d8fff4",
            troughcolor="#003a3a",
            highlightthickness=0,
            bd=0,
            sliderrelief="flat",
            font=("Consolas", 8),
        )
        scale.pack(side="left", fill="x", expand=True)
        apt_vars[apt_name] = var
        pro_scales[apt_name] = scale

    tk.Label(
        pro_frame,
        text="Voice: flight mode · demo studio · accuracy 90 · increase wit. Click PRO again to cycle Flight ↔ studio demo.",
        fg="#66aa99",
        bg="#00181c",
        font=("Consolas", 8),
        anchor="w",
        wraplength=880,
        justify="left",
    ).pack(fill="x", padx=4, pady=(4, 2))

    crew_frame = tk.Frame(dock, bg="#00181c")
    tk.Label(
        crew_frame,
        text="GROK BOT CREW  —  Gmail · Calendar · Drive · GitHub · Stripe",
        fg="#ffe066",
        bg="#00181c",
        font=("Consolas", 8, "bold"),
        anchor="w",
    ).pack(fill="x", padx=4, pady=(2, 2))

    def launch_crew(job: str):
        handoff_bot(job)
        if _host:
            names = dict(BOT_CREW)
            label = names.get(job, job)
            _host.speak(f"{label} brief is on the clipboard. Paste it into Grok Bot.")
            _host.launch_site(_bot_url())

    crew_row = tk.Frame(crew_frame, bg="#00181c")
    crew_row.pack(fill="x", padx=4, pady=(0, 4))
    for job, lab in BOT_CREW:
        tk.Button(
            crew_row,
            text=lab,
            command=lambda j=job: launch_crew(j),
            bg="#003a3a",
            fg="#00ffd2",
            activebackground="#00ffd2",
            activeforeground="#001114",
            relief="flat",
            font=("Consolas", 8, "bold"),
            padx=8,
            pady=2,
        ).pack(side="left", padx=3)

    def _xy(h: int) -> Tuple[int, int]:
        try:
            if dock.winfo_ismapped():
                x0, y0 = int(dock.winfo_x()), int(dock.winfo_y())
            else:
                x0, y0 = pos["x"], pos["y"]
        except Exception:
            x0, y0 = pos["x"], pos["y"]
        x0, y0 = _clamp_xy(x0, y0, width, h, sw, sh)
        pos["x"], pos["y"] = x0, y0
        return x0, y0

    def size_for():
        if expanded == "unlock":
            h = pro_h
        elif expanded:
            h = expanded_h
        else:
            h = collapsed
        px, py = _xy(h)
        dock.geometry(f"{width}x{h}+{px}+{py}")
        if expanded == "unlock":
            body.pack_forget()
            pro_frame.pack(fill="both", expand=True, padx=8, pady=(0, 6))
            refresh_pro(force_scales=True)
        elif expanded:
            pro_frame.pack_forget()
            body.pack(fill="both", expand=True, padx=8, pady=(0, 6))
            _pro_ready["on"] = False
        else:
            body.pack_forget()
            pro_frame.pack_forget()
            _pro_ready["on"] = False

    def _sync_pro_tile():
        import sylph_license

        btn = btns.get("unlock")
        if not btn:
            return
        if sylph_license.is_premium():
            btn.configure(text="FLIGHT")
        elif sylph_license.is_licensed():
            btn.configure(text="DEMO")
        else:
            btn.configure(text="PRO")

    def refresh_pro(force_scales: bool = False):
        import sylph_license

        licensed = sylph_license.is_licensed()
        a = sylph_license.aptitudes()
        header_var.set(sylph_license.status_line() if licensed else sylph_license.PRICE_PITCH)
        _pro_armed["on"] = False
        if force_scales or not _pro_ready["on"]:
            for name, var in apt_vars.items():
                try:
                    if int(var.get()) != a[name]:
                        var.set(a[name])
                except Exception:
                    var.set(a[name])
            _pro_ready["on"] = True
        _pro_armed["on"] = True
        state = "normal" if licensed else "disabled"
        for scale in pro_scales.values():
            scale.configure(state=state)
        for child in pro_btns.winfo_children():
            child.configure(state=state)
        _sync_pro_tile()

    def paint():
        if expanded == "unlock":
            refresh_pro()
            return
        body.configure(state="normal")
        body.delete("1.0", "end")
        if expanded == "gpu":
            body.insert("end", "GPU SESSION\n", "H")
            if gpu_hist:
                load, temp = gpu_hist[-1]
                body.insert("end", f"Load {load:.0f}%   Temp {temp:.0f}°C\n")
                spark = "".join("▁▂▃▄▅▆▇"[min(6, int(p[0] / 15))] for p in list(gpu_hist)[-40:])
                body.insert("end", spark + "\n", "F")
            else:
                body.insert("end", "Sampling…\n")
            occ = gpu_occupancy_snap or None
            if occ:
                gpu = occ.get("gpu") or {}
                body.insert(
                    "end",
                    f"Board {gpu.get('used_mib')} / {gpu.get('total_mib')} MiB   "
                    f"free {gpu.get('free_mib')}   {gpu.get('name','')}\n",
                )
                models = occ.get("ollama_models") or []
                body.insert(
                    "end",
                    "LLM  " + (", ".join(m.get("name") for m in models) if models else "none") + "\n",
                )
                games = occ.get("games") or []
                body.insert(
                    "end",
                    "GAME " + (", ".join(g.get("label") for g in games) if games else "none") + "\n",
                )
                body.insert("end", "WHO  (dedicated MiB, DWM is compositor)\n", "H")
                for p in (occ.get("top") or [])[:8]:
                    body.insert(
                        "end",
                        f"  {p.get('dedicated_mib'):7.0f}  {p.get('kind'):14}  {p.get('label')}\n",
                    )
        elif expanded == "wx":
            body.insert("end", "WEATHER + AQI  (Open-Meteo)\n", "H")
            body.insert("end", weather_line() + "\n")
        elif expanded == "cal":
            body.insert("end", "CALENDAR\n", "H")
            body.insert("end", calendar_line() + "\n")
            for e in events[:6]:
                body.insert("end", f"  {_ics_when(e.get('start',''))}  {e.get('summary','')}\n")
        elif expanded == "focus":
            body.insert("end", "FOCUS / POMODORO\n", "H")
            if focus_until > time.time():
                left = int(focus_until - time.time())
                body.insert("end", f"Remaining {left // 60:02d}:{left % 60:02d}\n")
            else:
                body.insert("end", f"Idle. Say sylph start focus, or {focus_minutes} minutes.\n")
        elif expanded == "media":
            body.insert("end", "NOW PLAYING / MIXER\n", "H")
            body.insert("end", "\n".join(media) or "No active audio sessions.\n")
        elif expanded == "clip":
            body.insert("end", "CLIPBOARD VAULT\n", "H")
            if not clipboard:
                body.insert("end", "Copy something. I'll keep the last 20 clips locally.\n")
            for i, item in enumerate(clipboard[:8], 1):
                body.insert("end", f"{i}. {item[:180].replace(chr(10), ' ')}\n")
        elif expanded == "home":
            body.insert("end", f"HOME  —  {home_env.upper()}\n", "H")
            if home_env == "ha":
                if not ha_tiles:
                    body.insert("end", "Set HA_URL and HA_KEY in config.json for live tiles.\n", "F")
                else:
                    for t in ha_tiles[:8]:
                        body.insert("end", f"  {t.get('name')}  [{t.get('state')}]\n")
                    body.insert("end", "\nSay sylph lights on, or click HA tiles via voice.\n", "F")
            elif home_env == "google":
                body.insert("end", "Google Home: opens your Google Home app/page. Commands route through Home Assistant if configured.\n")
                body.insert("end", "Say sylph google turn on the kitchen lights.\n", "F")
            else:
                body.insert("end", "Amazon Alexa: opens your Alexa app/page. Commands route through Home Assistant if configured.\n")
                body.insert("end", "Say sylph alexa set the thermostat to 70.\n", "F")
        elif expanded == "links":
            body.insert("end", "LINK SHELF\n", "H")
            if not links:
                body.insert("end", "Paste a URL in the console. I'll shelf it.\n")
            for item in reversed(links[-8:]):
                body.insert("end", f"  {item.get('ts','')}  {item.get('title') or item.get('url')}\n")
        elif expanded == "quiet":
            body.insert("end", "QUIET HOURS\n", "H")
            body.insert("end", "ON — mic ignores me except 'sylph I'm back'.\n" if quiet_hours else "OFF — I'm listening for sylph.\n")
        elif expanded == "shot":
            body.insert("end", "SCREENSHOT → ASK\n", "H")
            body.insert("end", "Captures the desktop and sends it to me as an image.\nSay sylph screenshot and ask, or click SHOT.\n")
        elif expanded == "bot":
            body.insert("end", "GROK BOT  —  xAI TEAMMATE\n", "H")
            body.insert("end", "I stay on this desktop. Grok Bot gets its own cloud computer.\n")
            body.insert("end", "Crew (OAuth in the Bot app — click a button above):\n")
            body.insert("end", "  GMAIL  Calendar  Drive  GitHub  Stripe\n")
            body.insert("end", "Gmail/Calendar/Drive/Stripe: connect cards already Added.\n")
            body.insert("end", "GitHub: still needs a PAT in Grok Bot (not in SYLPH).\n", "F")
            body.insert("end", "Also: MAIL tile · watch/home/wx handoff. Say open chrome.\n")
            body.insert("end", _bot_url() + "\n", "F")
        elif expanded == "mail":
            body.insert("end", "GMAIL  —  GROK BOT\n", "H")
            body.insert("end", f"Account: {_mail_account() or '(set mail_account in local owner.json)'}\n")
            body.insert("end", "Brief on clipboard. Gmail connector is Added in Grok Bot.\n")
            body.insert("end", "Drafts only. Never send.\n", "F")
        elif expanded == "gcal":
            body.insert("end", "GOOGLE CALENDAR  —  GROK BOT\n", "H")
            body.insert("end", "Uses the Google Calendar connector (Added).\n")
            body.insert("end", "ICS tile CAL is local; this tile is the Bot.\n", "F")
        elif expanded == "drive":
            body.insert("end", "GOOGLE DRIVE  —  GROK BOT\n", "H")
            body.insert("end", "Drive connector is Added. Find / summarize / organize.\n")
            body.insert("end", "No deletes or external shares unless you said so.\n", "F")
        elif expanded == "github":
            body.insert("end", "GITHUB  —  GROK BOT\n", "H")
            body.insert("end", "GitHub still needs a PAT in the Bot app, not in SYLPH.\n")
            body.insert("end", "github.com/settings/tokens → classic → repo + read:org.\n", "F")
        elif expanded == "stripe":
            body.insert("end", "STRIPE  —  GROK BOT\n", "H")
            body.insert("end", "Stripe connector is Added. Products, prices, customers.\n")
            body.insert("end", "No live charges unless you said so this turn.\n", "F")
        body.configure(state="disabled")

    def select(name: str):
        global expanded, quiet_hours, home_env
        if name == "quiet":
            msg = set_quiet(not quiet_hours)
            if _host:
                _host.speak(msg)
            expanded = "quiet"
        elif name == "shot":
            expanded = "shot"
            size_for()
            paint()
            if _host:
                Thread(target=_host.screenshot_ask, daemon=True).start()
            return
        elif name == "bot":
            expanded = "bot"
            size_for()
            paint()
            if _host:
                _host.speak("Opening Grok Bot. That's the xAI teammate with its own computer.")
                _host.launch_site(_bot_url())
            return
        elif name in ("mail", "gcal", "drive", "github", "stripe"):
            job = {"mail": "mail", "gcal": "cal", "drive": "drive", "github": "github", "stripe": "stripe"}[name]
            expanded = name
            size_for()
            paint()
            launch_crew(job)
            return
        elif name == "focus":
            expanded = "focus"
            if focus_until <= time.time() and _host:
                _host.speak(start_focus(25))
        elif name == "unlock":
            import sylph_license

            if expanded == "unlock" and sylph_license.is_licensed():
                msg = sylph_license.toggle_demo()
                if _host:
                    _host.speak(msg)
                    _host.chat_note("SYS", msg)
                refresh_pro(force_scales=True)
                return
            expanded = "unlock"
            size_for()
            paint()
            if _host:
                _host.speak(
                    sylph_license.status_line()
                    if sylph_license.is_licensed()
                    else "Studio edition. Unlock Flight for fifty dollars once, or eight a month. Paste a key: unlock, then your key."
                )
            return
        elif name == "home":
            # cycle HA -> google -> alexa on repeated clicks when already open
            if expanded == "home":
                home_env = {"ha": "google", "google": "alexa", "alexa": "ha"}.get(home_env, "ha")
            expanded = "home"
        else:
            expanded = name if expanded != name else ""
        size_for()
        paint()
        Thread(target=_prefetch, daemon=True).start()

    def _prefetch():
        if expanded == "wx":
            weather_refresh(True)
        elif expanded == "cal":
            calendar_refresh(True)
        elif expanded == "media":
            media_poll()
        elif expanded == "home" and home_env == "ha":
            ha_poll(True)
        mark()

    for key, label in TILES:
        b = tk.Button(
            bar,
            text=label,
            command=lambda k=key: select(k),
            bg="#003a3a",
            fg="#00ffd2",
            activebackground="#00ffd2",
            activeforeground="#001114",
            relief="flat",
            font=("Consolas", 8, "bold"),
            padx=5,
            pady=2,
        )
        b.pack(side="left", padx=2)
        btns[key] = b

    def home_pick(env: str):
        global home_env, expanded
        home_env = env
        expanded = "home"
        size_for()
        paint()
        if env == "google" and _host:
            _host.launch_site("https://home.google.com")
        elif env == "alexa" and _host:
            _host.launch_site("https://alexa.amazon.com")
        elif env == "ha":
            Thread(target=lambda: ha_poll(True), daemon=True).start()

    sub = tk.Frame(bar, bg=_bg())
    sub.pack(side="right")
    for env, lab in (("ha", "HA"), ("google", "GGL"), ("alexa", "ALX")):
        tk.Button(
            sub,
            text=lab,
            command=lambda e=env: home_pick(e),
            bg="#002828",
            fg="#7af0ff",
            relief="flat",
            font=("Consolas", 7, "bold"),
            padx=4,
        ).pack(side="left", padx=1)

    def _bind_move(handle):
        drag = {"x": 0, "y": 0}

        def start(event):
            drag["x"], drag["y"] = event.x_root, event.y_root

        def motion(event):
            dx = event.x_root - drag["x"]
            dy = event.y_root - drag["y"]
            drag["x"], drag["y"] = event.x_root, event.y_root
            try:
                nx, ny = _clamp_xy(
                    dock.winfo_x() + dx,
                    dock.winfo_y() + dy,
                    dock.winfo_width() or width,
                    dock.winfo_height() or collapsed,
                    sw,
                    sh,
                )
            except Exception:
                nx, ny = pos["x"] + dx, pos["y"] + dy
            pos["x"], pos["y"] = nx, ny
            dock.geometry(f"+{nx}+{ny}")

        def stop(_event=None):
            _save_layout(pos["x"], pos["y"])

        handle.bind("<ButtonPress-1>", start)
        handle.bind("<B1-Motion>", motion)
        handle.bind("<ButtonRelease-1>", stop)

    def reset_pos(_event=None):
        pos["x"] = max(8, (sw - width) // 2)
        pos["y"] = 8
        size_for()
        _save_layout(pos["x"], pos["y"])
        if _host and _event is not None:
            _host.speak("Desk bar parked back at the top.")
        return "break"

    _bind_move(title)
    _bind_move(bar)
    title.bind("<Double-Button-1>", reset_pos)
    _reset_dock = lambda: reset_pos()

    _paint = paint
    sample_gpu()
    _sync_pro_tile()
    paint()
    logger.info("Utility dock armed at %sx%s (drag ☰ SYLPH DESK)", pos["x"], pos["y"])
    return dock


def handle_desk_intent(t: str) -> Optional[str]:
    global home_env, expanded
    low = (t or "").lower()
    if any(
        k in low
        for k in ("reset desk", "park desk", "desk default", "unstick desk", "move desk back", "desk to the top")
    ):
        if _reset_dock:
            _reset_dock()
            return "Desk bar is back at the top. Drag the SYLPH DESK handle to park it anywhere."
        return "Desk isn't on screen yet."
    if any(k in low for k in ("quiet hours", "i'm recording", "im recording", "be quiet", "go quiet")):
        return set_quiet(True)
    if is_quiet_wake(low):
        return set_quiet(False)
    if "cancel focus" in low or "end pomodoro" in low or "stop focus" in low:
        return cancel_focus()
    if "pomodoro" in low or "start focus" in low or "focus session" in low:
        m = re.search(r"(\d{1,3})", low)
        return start_focus(int(m.group(1)) if m else 25)
    if any(k in low for k in ("weather", "aqi", "air quality", "how hot outside", "temperature outside")):
        weather_refresh(True)
        return weather_line()
    if any(k in low for k in ("calendar", "what's next", "whats next", "my meetings", "next event")):
        calendar_refresh(True)
        return calendar_line()
    if "clipboard" in low or "what did i copy" in low:
        clip_poll()
        return ("Clipboard: " + clipboard[0][:220]) if clipboard else "Clipboard empty."
    if any(k in low for k in ("what's playing", "whats playing", "now playing", "audio sessions")):
        media_poll()
        return "Media: " + ("; ".join(media[:4]) if media else "nothing loud.")
    if low.startswith("unlock sylph-") or low.startswith("activate license"):
        key = re.sub(r"^(unlock|activate license)\s+", "", t.strip(), flags=re.I)
        import sylph_license

        msg = sylph_license.save_license(key, "owner")
        mark()
        return msg
    if any(
        k in low
        for k in ("flight mode", "full grok", "smart mode", "smart sylph", "restore flight")
    ):
        import sylph_license

        msg = sylph_license.preset_flight()
        mark()
        return msg
    if any(
        k in low
        for k in ("demo studio", "studio demo", "dumb mode", "studio edition", "demo the dumb")
    ):
        import sylph_license

        msg = sylph_license.preset_studio_demo()
        mark()
        return msg
    m = re.search(r"(increase|decrease|raise|lower|boost|drop)\s+(accuracy|wit|depth|spoken)", low)
    if m:
        import sylph_license

        delta = 10 if m.group(1) in ("increase", "raise", "boost") else -10
        msg = sylph_license.nudge_aptitude(m.group(2), delta)
        mark()
        return msg
    m = re.search(r"\b(accuracy|wit|depth|spoken)\s*(?:to|=|:)?\s*(\d{1,3})\b", low)
    if m:
        import sylph_license

        msg = sylph_license.set_aptitude(m.group(1), int(m.group(2)))
        mark()
        return msg
    if any(k in low for k in ("aptitude", "what mode", "sylph mode", "license status")):
        import sylph_license

        return sylph_license.status_line()
    if "screenshot" in low and any(k in low for k in ("ask", "what is this", "what's this", "explain", "read")):
        return "SHOT"
    if re.search(r"\b(open chrome|launch chrome|new chrome window|chrome window)\b", low):
        url = ""
        m = re.search(r"https?://[^\s<>\"']+", t, re.I)
        if m:
            url = m.group(0)
        else:
            m = re.search(
                r"(?:chrome|window|tab)\s+(?:to|at|and\s+open)?\s*(.+)$",
                t.strip(),
                flags=re.I,
            )
            if m:
                url = m.group(1).strip()
                url = re.sub(r"^(to|at|open|launch)\s+", "", url, flags=re.I).strip()
        return "CHROME " + url
    if any(
        k in low
        for k in ("open grok.com", "open grok chat", "open grok in chrome")
    ):
        return "CHROME https://grok.com"
    if any(
        k in low
        for k in ("open grok bot", "launch grok bot", "hand off to grok bot", "open grokbot", "start grok bot")
    ):
        expanded = "bot"
        mark()
        return "GROK_BOT"
    if any(k in low for k in ("hand watch to grok", "watch bot", "grok bot watch")):
        expanded = "bot"
        return handoff_bot("watch", t.strip())
    if any(k in low for k in ("hand calendar to grok", "calendar bot", "cal bot", "grok bot calendar")):
        expanded = "cal"
        return handoff_bot("cal", t.strip())
    if any(k in low for k in ("hand home to grok", "home bot", "grok bot home")):
        expanded = "home"
        return handoff_bot("home", t.strip())
    if any(k in low for k in ("hand weather to grok", "weather bot", "wx bot", "grok bot weather")):
        expanded = "wx"
        return handoff_bot("wx", t.strip())
    if any(k in low for k in ("hand drive to grok", "drive bot", "google drive", "gdrive bot", "grok bot drive")):
        expanded = "bot"
        return handoff_bot("drive", t.strip())
    if any(k in low for k in ("hand github to grok", "github bot", "git hub bot", "grok bot github")):
        expanded = "bot"
        return handoff_bot("github", t.strip())
    if any(k in low for k in ("hand stripe to grok", "stripe bot", "pay bot", "grok bot stripe")):
        expanded = "bot"
        return handoff_bot("stripe", t.strip())
    if any(k in low for k in ("hand gmail to grok", "gmail bot", "google mail")):
        extra = t.strip()
        set_clipboard(mail_brief(extra))
        expanded = "mail"
        mark()
        return "GROK_MAIL"
    if any(
        k in low
        for k in (
            "check email",
            "check my email",
            "read my email",
            "read my mail",
            "check inbox",
            "my inbox",
            "draft a reply",
            "draft reply",
            "mail bot",
            "open mail bot",
        )
    ):
        extra = t.strip()
        set_clipboard(mail_brief(extra))
        expanded = "mail"
        mark()
        return "GROK_MAIL"
    if any(k in low for k in ("google home", "open google home")):
        home_env = "google"
        expanded = "home"
        mark()
        return "GOOGLE_HOME"
    if any(k in low for k in ("open alexa", "amazon alexa", "alexa app")):
        home_env = "alexa"
        expanded = "home"
        mark()
        return "ALEXA_HOME"
    if "home assistant" in low or (low.strip() == "home tiles"):
        home_env = "ha"
        expanded = "home"
        mark()
        return "Home Assistant tiles. Set HA_URL if they're empty."
    return None
