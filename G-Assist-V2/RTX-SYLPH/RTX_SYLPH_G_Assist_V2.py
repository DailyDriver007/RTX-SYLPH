"""
Copyright (c) 2025–2026 Kyle Baker. All rights reserved.
Personal use only. See LICENSE. No commercial use, public display, or re-skin
without a written license from Kyle Baker.

RTX SYLPH — G-Assist V2 Plugin
Supreme Secure Home Domination Edition
Original concept: @BanditsOfBedlam [Discord] / DailyDriver007 [GitHub]
Powered by Ara @ Colossus Data Center
"""

import os
import sys
import io
import base64
import logging
import json
import time
import math
import random
import subprocess
import html as html_lib
import re
import tempfile
import mimetypes
from urllib.parse import quote
from datetime import datetime, timezone, timedelta
from pathlib import Path
from queue import Empty, Queue
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None  # type: ignore

import cv2
import pygame
import numpy as np
import pyttsx3
import speech_recognition as sr
import GPUtil
import requests
from PIL import Image, ImageDraw, ImageFont
import bcrypt
import screeninfo

# ---------------------------------------------------------------------------
# Paths / logging
# ---------------------------------------------------------------------------
_plugin_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_plugin_dir)
_libs_path = os.path.join(_plugin_dir, "libs")
if os.path.exists(_libs_path) and _libs_path not in sys.path:
    sys.path.insert(0, _libs_path)

from gassist_sdk import Plugin
import sylph_desk
import sylph_license

logger = logging.getLogger("rtx_sylph")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    _file = logging.FileHandler(os.path.join(_plugin_dir, "sylph.log"), encoding="utf-8")
    _file.setFormatter(_fmt)
    _err = logging.StreamHandler(sys.stderr)
    _err.setFormatter(_fmt)
    logger.addHandler(_file)
    logger.addHandler(_err)
    logger.propagate = False

try:
    import faulthandler

    _crash_log = open(os.path.join(_plugin_dir, "sylph_crash.log"), "a", encoding="utf-8")
    faulthandler.enable(file=_crash_log, all_threads=True)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
config_path = os.path.join(_plugin_dir, "config.json")
if not os.path.isfile(config_path):
    raise FileNotFoundError(f"config.json missing next to the plugin: {config_path}")

with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)


def _cfg(key: str, default=""):
    value = config.get(key, default)
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip()
    return value


def _looks_like_placeholder(value: str) -> bool:
    if not value:
        return True
    lowered = value.lower()
    needles = (
        "your-",
        "your_",
        "example.com",
        "changeme",
        "placeholder",
        "home-assistant-ip",
    )
    return any(n in lowered for n in needles)


def get_secret(secret_name: str) -> Optional[str]:
    """Optional AWS Secrets Manager lookup. Off by default so startup is not blocked."""
    if not bool(config.get("USE_AWS_SECRETS", False)):
        return None
    try:
        import boto3

        client = boto3.client(
            "secretsmanager",
            region_name=_cfg("AWS_REGION", "us-east-2") or "us-east-2",
        )
        return client.get_secret_value(SecretId=secret_name)["SecretString"]
    except Exception as e:
        logger.warning("Secret %s fetch skipped: %s", secret_name, e)
        return None


def _resolve_key(secret_id: str, config_key: str) -> str:
    from_config = _cfg(config_key, "")
    if from_config and not _looks_like_placeholder(from_config):
        return from_config
    from_aws = get_secret(secret_id)
    return (from_aws or "").strip()


GROK_API_KEY = _resolve_key("grok-api-key", "GROK_API_KEY")
OPENAI_API_KEY = _resolve_key("openai-api-key", "OPENAI_API_KEY")
NVIDIA_API_KEY = _resolve_key("nvidia-api-key", "NVIDIA_API_KEY")
HUGGINGFACE_API_KEY = _resolve_key("huggingface-api-key", "HUGGINGFACE_API_KEY")
DEEPINFRA_API_KEY = _resolve_key("deepinfra-api-key", "DEEPINFRA_API_KEY")
MISTRAL_API_KEY = _resolve_key("mistral-api-key", "MISTRAL_API_KEY")
HA_URL = "" if _looks_like_placeholder(_cfg("HA_URL", "")) else _cfg("HA_URL", "").rstrip("/")
HA_KEY = _resolve_key("ha-key", "HA_KEY")
PLAIN_PIN = _cfg("PIN", "1234") or "1234"
HASHED_PIN = bcrypt.hashpw(PLAIN_PIN.encode("utf-8"), bcrypt.gensalt())
WAKE_WORD = (_cfg("WAKE_WORD", "sylph") or "sylph").lower()
VOICE_SPEED = int(config.get("VOICE_SPEED", 165) or 165)
VOICE_NAME = (_cfg("VOICE_NAME", "ara") or "ara").lower()
VOICE_ENGINE = (_cfg("VOICE_ENGINE", "xai") or "xai").lower()
GROK_MODEL = _cfg("GROK_MODEL", "grok-4") or "grok-4"
NEMOTRON_LIGHTNING_MODEL = (
    _cfg("NEMOTRON_LIGHTNING_MODEL", "")
    or _cfg("NEMOTRON_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
    or "nvidia/nemotron-3.5-lightning-30b-a3b"
)
NEMOTRON_SUPER_MODEL = _cfg("NEMOTRON_SUPER_MODEL", "nvidia/nemotron-3-super-120b-a12b") or "nvidia/nemotron-3-super-120b-a12b"
NEMOTRON_ULTRA_MODEL = _cfg("NEMOTRON_ULTRA_MODEL", "nvidia/llama-3.1-nemotron-ultra-253b-v1") or "nvidia/llama-3.1-nemotron-ultra-253b-v1"
NEMOTRON_MODEL = NEMOTRON_LIGHTNING_MODEL
GEMINI_API_KEY = _resolve_key("gemini-api-key", "GEMINI_API_KEY")
GEMINI_FLASH_MODEL = (
    _cfg("GEMINI_FLASH_MODEL", "")
    or _cfg("GEMINI_MODEL", "gemini-3.7-flash")
    or "gemini-3.7-flash"
)
GEMINI_PRO_MODEL = _cfg("GEMINI_PRO_MODEL", "gemini-3.1-pro-preview") or "gemini-3.1-pro-preview"
GEMINI_MODEL = GEMINI_FLASH_MODEL
CLOCK_ENABLED = bool(config.get("CLOCK_ENABLED", True))
CHAT_ENABLED = bool(config.get("CHAT_ENABLED", True))
WATCH_ENABLED = bool(config.get("WATCH_ENABLED", True))
DOCK_ENABLED = bool(config.get("DOCK_ENABLED", True))
CALENDAR_ICS_URL = _cfg("CALENDAR_ICS_URL", "")
WEATHER_LAT = _cfg("WEATHER_LAT", "")
WEATHER_LON = _cfg("WEATHER_LON", "")
SMART_HOME_ENV = (_cfg("SMART_HOME_ENV", "ha") or "ha").lower()
sylph_license.load_license(
    _cfg("SYLPH_LICENSE_KEY", ""),
    owner_flight=bool(config.get("SYLPH_OWNER_FLIGHT", False)),
)
GROK_BOT_URL = _cfg("GROK_BOT_URL", "https://x.ai/bot") or "https://x.ai/bot"
TMDB_API_KEY = _resolve_key("tmdb-api-key", "TMDB_API_KEY")
TMDB_READ_TOKEN = _resolve_key("tmdb-read-token", "TMDB_READ_TOKEN")
WATCH_REGION = (_cfg("WATCH_REGION", "US") or "US").upper()
TMDB_ATTRIBUTION = "This product uses the TMDB API but is not endorsed or certified by TMDB."
XAI_TTS_VOICES = {"ara", "eve", "leo", "rex", "sal"}
CAMERA_URL = _cfg("CAMERA_URL", "")
VIDEO_PATH = _cfg("VIDEO_PATH", "assets/rtx_sylph_animated.mp4")
FALLBACK_IMAGE_PATH = _cfg("FALLBACK_IMAGE_PATH", "assets/SYLPH_Icon.png")
MIC_DEVICE = _cfg("MIC_DEVICE", "QuadCast")
WAKE_ALIASES = {
    "sylph", "silph", "sylf", "silf", "self", "sylphs", "sylphe",
    "silk", "sylph.", "slyph",
}


def resolve_browser() -> str:
    candidates = [
        _cfg("BROWSER_PATH", ""),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return "chrome"


BROWSER_PATH = resolve_browser()

# ---------------------------------------------------------------------------
# Plugin (created before background threads so wake/stream can see it)
# ---------------------------------------------------------------------------
plugin = Plugin(
    name="RTX SYLPH",
    version="8.4",
    description="Supreme Secure Home Domination Edition",
)

# ---------------------------------------------------------------------------
# Display sizing (inches -> pixels at the real Windows DPI)
# ---------------------------------------------------------------------------
def screen_dpi() -> int:
    if sys.platform != "win32":
        return 96
    try:
        import ctypes

        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
        dpi = int(ctypes.windll.user32.GetDpiForSystem() or 0)
        return dpi if dpi >= 72 else 96
    except Exception:
        return 96


def inches_px(inches: float) -> int:
    return max(80, int(round(float(inches) * screen_dpi())))


# ---------------------------------------------------------------------------
# Voice — Ara via xAI TTS when possible, SAPI fallback. One worker thread.
# ---------------------------------------------------------------------------
engine = None
_speech_q: Queue = Queue()
_tts_cache: Dict[str, bytes] = {}
_mixer_ready = False
_clock_root = None
_panels_thread = None


def _xai_tts_bytes(text: str) -> Optional[bytes]:
    """Ara (or eve/leo/rex/sal) through xAI TTS. Returns mp3 bytes or None."""
    if not GROK_API_KEY or not text:
        return None
    cached = _tts_cache.get(text)
    if cached:
        return cached
    voice_id = VOICE_NAME if VOICE_NAME in XAI_TTS_VOICES else "ara"
    try:
        resp = requests.post(
            "https://api.x.ai/v1/tts",
            headers={"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"},
            json={
                "text": text[:12000],
                "voice_id": voice_id,
                "language": "en",
                "speed": 1.08,
                "output_format": {"codec": "mp3", "sample_rate": 24000, "bit_rate": 128000},
            },
            timeout=20,
        )
        if resp.status_code >= 400:
            logger.warning("xAI TTS HTTP %s: %s", resp.status_code, (resp.text or "")[:280])
            return None
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if "json" in ctype or "text" in ctype:
            logger.warning("xAI TTS unexpected payload: %s", (resp.text or "")[:280])
            return None
        audio = resp.content
        if not audio or len(audio) < 80:
            return None
        if len(text) < 80:
            _tts_cache[text] = audio
        return audio
    except Exception as e:
        logger.warning("xAI TTS failed: %s", e)
        return None


def _play_mp3_bytes(audio: bytes) -> bool:
    global _mixer_ready
    if not audio:
        return False
    path = None
    try:
        if not _mixer_ready:
            pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=512)
            _mixer_ready = True
        fd, path = tempfile.mkstemp(suffix=".mp3", prefix="sylph_tts_")
        os.close(fd)
        with open(path, "wb") as handle:
            handle.write(audio)
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy() and not shutdown_flag:
            time.sleep(0.04)
        try:
            pygame.mixer.music.unload()
        except Exception:
            pass
        return True
    except Exception as e:
        logger.warning("TTS playback failed: %s", e)
        return False
    finally:
        if path:
            try:
                os.remove(path)
            except Exception:
                pass


def voice_worker():
    """Own speech for the life of the process. Prefer Ara; fall back to SAPI."""
    global engine, _mixer_ready
    inst = None
    try:
        inst = pyttsx3.init()
        inst.setProperty("rate", VOICE_SPEED)
        wanted = VOICE_NAME
        picked = None
        for voice in inst.getProperty("voices") or []:
            blob = f"{getattr(voice, 'name', '')} {getattr(voice, 'id', '')}".lower()
            if wanted in blob or any(n in blob for n in ("zira", "jenny", "aria", "female")):
                if wanted in blob:
                    picked = voice
                    break
                if picked is None:
                    picked = voice
        if picked is not None:
            inst.setProperty("voice", picked.id)
            logger.info("SAPI fallback voice: %s", picked.name)
        engine = inst
        logger.info("Voice worker ready (engine=%s voice=%s)", VOICE_ENGINE, VOICE_NAME)
    except Exception as e:
        logger.error("Voice engine init failed: %s", e)
        engine = None
    use_xai = VOICE_ENGINE in ("xai", "ara", "grok") and bool(GROK_API_KEY)
    if inst is None and not use_xai:
        return
    while not shutdown_flag:
        try:
            text = _speech_q.get(timeout=0.2)
        except Empty:
            continue
        if text is None:
            break
        spoken = False
        if use_xai:
            audio = _xai_tts_bytes(text)
            if audio:
                spoken = _play_mp3_bytes(audio)
        if not spoken and inst is not None:
            try:
                inst.say(text)
                inst.runAndWait()
            except Exception as e:
                logger.error("Voice output failed: %s", e)


def init_voice():
    Thread(target=voice_worker, daemon=True, name="sylph-voice").start()
    deadline = time.time() + 6
    while engine is None and time.time() < deadline and not shutdown_flag:
        time.sleep(0.05)
    if engine is None:
        logger.warning("Voice worker not ready yet — speech will start when it is")


def speak(text: str):
    """Non-blocking. The wake thread must never wait on TTS."""
    if not text:
        return
    logger.info("SYLPH: %s", text)
    try:
        _speech_q.put_nowait(text)
    except Exception as e:
        logger.error("Speech queue failed: %s", e)


# ---------------------------------------------------------------------------
# Avatar / animation
# ---------------------------------------------------------------------------
AVATAR_W, AVATAR_H = inches_px(3), inches_px(4)
COUNCIL_PX = min(340, inches_px(3.2))
hud_x, hud_y = 0, 0
hud_user_placed = False
video_lock = Lock()
current_state = "idle"
current_video_index = 0
playback_mode = "sequential"
forced_video = None
forced_end_time = 0.0
state_hold_until = 0.0
state_sticky = False
pending_state: Optional[str] = None
pending_hold: Optional[float] = None
pending_reload = False
cap_current = None
_hud_locked = False
last_surface = None
last_frame_np = None
last_src_wh = (0, 0)
fallback_surface = None
screen = None
runtime_started = False
shutdown_flag = False
ANIMATION_CLASSES: Dict[str, List[str]] = {}
_spawned_procs: List[subprocess.Popen] = []

# Mood / "states of being" — each maps to a Drive clip class.
# sticky states stay until another set_state(); others expire back to idle.
STATE_META = {
    "idle":         {"hold": 0,  "next": "idle",        "sticky": False, "overlay": None},
    "listening":    {"hold": 12, "next": "idle",        "sticky": False, "overlay": None},
    "thinking":     {"hold": 0,  "next": "thinking",    "sticky": True,  "overlay": None},
    "answering":    {"hold": 10, "next": "idle",        "sticky": False, "overlay": None},
    "gpu_cool":     {"hold": 14, "next": "idle",        "sticky": False, "overlay": None},
    "gpu_overheat": {"hold": 14, "next": "idle",        "sticky": False, "overlay": None},
    "home_assist":  {"hold": 10, "next": "idle",        "sticky": False, "overlay": None},
    "sound_system": {"hold": 10, "next": "idle",        "sticky": False, "overlay": None},
    "camera_mode":  {"hold": 10, "next": "idle",        "sticky": False, "overlay": None},
}


def _abs_asset(rel_or_abs: str) -> str:
    if not rel_or_abs:
        return ""
    if os.path.isabs(rel_or_abs) and os.path.isfile(rel_or_abs):
        return rel_or_abs
    candidate = os.path.join(_plugin_dir, rel_or_abs)
    if os.path.isfile(candidate):
        return candidate
    nested = os.path.join(_plugin_dir, "assets", "assets", os.path.basename(rel_or_abs))
    if os.path.isfile(nested):
        return nested
    return candidate


def _asset_dirs() -> List[str]:
    dirs = [
        os.path.join(_plugin_dir, "assets"),
        os.path.join(_plugin_dir, "assets", "assets"),
    ]
    return [d for d in dirs if os.path.isdir(d)]


def discover_videos(*prefixes: str) -> List[str]:
    found: List[str] = []
    seen = set()
    lowered = tuple(p.lower() for p in prefixes)
    for folder in _asset_dirs():
        try:
            names = os.listdir(folder)
        except OSError:
            continue
        for name in names:
            if not name.lower().endswith(".mp4"):
                continue
            stem = name[:-4].lower()
            if any(stem.startswith(prefix) for prefix in lowered):
                key = name.lower()
                if key in seen:
                    continue
                seen.add(key)
                found.append(os.path.join(folder, name))
    found.sort(key=lambda p: os.path.basename(p).lower())
    return found


def _unique_paths(*groups: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for group in groups:
        for path in group:
            key = os.path.normcase(path)
            if key in seen:
                continue
            seen.add(key)
            out.append(path)
    return out


def build_animation_classes() -> Dict[str, List[str]]:
    idle = discover_videos("sylph_idle", "rtx_sylph_animated")
    extra_idle = _abs_asset(VIDEO_PATH)
    if extra_idle and os.path.isfile(extra_idle) and extra_idle not in idle:
        idle.append(extra_idle)
    thinking = discover_videos("sylph_thinking")
    answering = discover_videos("sylph_answers")
    gpu_cool = discover_videos("gpu_cool")
    gpu_overheat = discover_videos("gpu_overheat")
    home_assist = discover_videos("sylph_home_assist")
    sound_system = discover_videos("rtx_sylph_sound_system")
    camera_mode = discover_videos("sylph_camera", "sylph_pc_rog")
    # Each pack is a state of being. Idle stays rest; thinking/answering/GPU/home
    # only play when that state is on. Every clip in a pack still cycles in order.
    fallback = idle[:] or thinking[:] or answering[:]
    classes = {
        "idle": idle,
        "listening": idle[:],
        "thinking": thinking,
        "answering": answering,
        "gpu_cool": gpu_cool,
        "gpu_overheat": gpu_overheat,
        "home_assist": home_assist,
        "sound_system": sound_system,
        "camera_mode": camera_mode,
    }
    for key, videos in list(classes.items()):
        if not videos:
            classes[key] = fallback[:]
            logger.warning("No videos for class '%s' — falling back to idle", key)
        else:
            logger.info("Animation class '%s': %d clips", key, len(videos))
    logger.info(
        "Library %d unique clips across being-states (idle %d, thinking %d, answering %d)",
        len(_unique_paths(*classes.values())),
        len(classes["idle"]),
        len(classes["thinking"]),
        len(classes["answering"]),
    )
    return classes


def size_hud_for_library(paths: List[str]) -> None:
    """Size the HUD once to the widest clip aspect so no source pixels are cropped.
    Taller/narrower clips letterbox. Window is larger so environments stay readable."""
    global AVATAR_W, AVATAR_H, _hud_locked
    max_aspect = 0.0
    src_w = src_h = 0
    seen = set()
    for path in paths:
        key = os.path.normcase(path)
        if key in seen:
            continue
        seen.add(key)
        cap = None
        try:
            cap = cv2.VideoCapture(path)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        except Exception:
            w = h = 0
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
        if w > 1 and h > 1:
            aspect = w / float(h)
            if aspect > max_aspect:
                max_aspect = aspect
                src_w, src_h = w, h
    if max_aspect <= 0:
        max_aspect = 9 / 16.0
        src_w, src_h = 416, 752
    sw, sh = screen_size()
    th = min(inches_px(8.5), int(sh * 0.82))
    tw = max(80, int(round(th * max_aspect)))
    if tw > int(sw * 0.48):
        tw = max(80, int(sw * 0.48))
        th = max(80, int(round(tw / max_aspect)))
    AVATAR_W, AVATAR_H = tw, th
    _hud_locked = True
    logger.info(
        "HUD %dx%d shows full frames (widest clip %dx%d aspect %.3f, %d files)",
        tw, th, src_w, src_h, max_aspect, len(seen),
    )


def _blank_surface():
    surf = pygame.Surface((AVATAR_W, AVATAR_H))
    surf.fill((0, 16, 16))
    return surf


def load_fallback_surface():
    global fallback_surface
    for candidate in (
        _abs_asset(FALLBACK_IMAGE_PATH),
        os.path.join(_plugin_dir, "assets", "SYLPH_Icon.png"),
        os.path.join(_plugin_dir, "assets", "assets", "SYLPH_Icon.png"),
    ):
        if candidate and os.path.isfile(candidate) and candidate.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            try:
                img = pygame.image.load(candidate)
                fallback_surface = pygame.transform.smoothscale(img, (AVATAR_W, AVATAR_H))
                return
            except Exception as e:
                logger.warning("Fallback image failed (%s): %s", candidate, e)
    fallback_surface = _blank_surface()


def load_next_video():
    global cap_current, current_video_index
    class_videos = ANIMATION_CLASSES.get(current_state) or ANIMATION_CLASSES.get("idle") or []
    candidates: List[str] = []

    if forced_video and time.time() < forced_end_time and os.path.isfile(forced_video):
        candidates = [forced_video]
    elif class_videos:
        if playback_mode == "random":
            candidates = class_videos[:]
            random.shuffle(candidates)
        else:
            start = current_video_index % len(class_videos)
            candidates = class_videos[start:] + class_videos[:start]
            current_video_index = (start + 1) % len(class_videos)

    if cap_current is not None:
        try:
            cap_current.release()
        except Exception:
            pass
        cap_current = None

    for video_path in candidates:
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if cap is not None and cap.isOpened():
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                cap_current = cap
                logger.info("Loaded: %s (%s)", os.path.basename(video_path), current_state)
                return
            if cap is not None:
                cap.release()
        except Exception as e:
            logger.warning("Clip failed %s: %s", os.path.basename(video_path), e)
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    logger.warning("No playable video for state '%s'", current_state)


def _fit_frame(frame, tw: int, th: int):
    """Scale the entire source frame into tw x th. Never crop."""
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((th, tw, 3), dtype=np.uint8)
    scale = min(tw / w, th / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    if nw == tw and nh == th:
        return resized
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    x, y = (tw - nw) // 2, (th - nh) // 2
    canvas[y : y + nh, x : x + nw] = resized
    return canvas


def resize_hud_to_frame(frame_w: int, frame_h: int):
    """Library pass already sized the HUD. Do not recreate the pygame window per clip."""
    if _hud_locked or frame_w < 2 or frame_h < 2:
        return


def _hud_hwnd():
    if screen is None:
        return None
    try:
        return pygame.display.get_wm_info().get("window")
    except Exception:
        return None


def hud_center_pos() -> Tuple[int, int]:
    sw, sh = screen_size()
    return max(0, (sw - AVATAR_W) // 2), max(0, (sh - AVATAR_H) // 2)


def place_hud(x: Optional[int] = None, y: Optional[int] = None):
    """Move the frameless HUD. Default is screen center."""
    global hud_x, hud_y
    if x is None or y is None:
        if hud_user_placed:
            x, y = hud_x, hud_y
        else:
            x, y = hud_center_pos()
    sw, sh = screen_size()
    x = max(0, min(sw - AVATAR_W, int(x)))
    y = max(0, min(sh - AVATAR_H, int(y)))
    hud_x, hud_y = x, y
    if sys.platform != "win32":
        os.environ["SDL_VIDEO_WINDOW_POS"] = f"{x},{y}"
        return
    hwnd = _hud_hwnd()
    if not hwnd:
        return
    try:
        import ctypes

        HWND_TOPMOST = -1
        SWP_SHOWWINDOW = 0x0040
        ctypes.windll.user32.SetWindowPos(
            int(hwnd), HWND_TOPMOST, x, y, AVATAR_W, AVATAR_H, SWP_SHOWWINDOW
        )
    except Exception as e:
        logger.warning("Could not place HUD: %s", e)


def _pin_hud_topmost():
    place_hud()


def _apply_pending_state():
    """Avatar thread only — open/close VideoCapture here, never from wake/SAPI."""
    global pending_state, pending_hold, current_state, current_video_index
    global state_hold_until, state_sticky
    if pending_state is None:
        return
    new_state = pending_state
    hold = pending_hold
    pending_state = None
    pending_hold = None
    if new_state not in ANIMATION_CLASSES:
        new_state = "idle"
    meta = STATE_META.get(new_state, STATE_META["idle"])
    seconds = meta["hold"] if hold is None else float(hold)
    changed = current_state != new_state
    current_state = new_state
    state_sticky = bool(meta.get("sticky"))
    state_hold_until = time.time() + seconds if seconds > 0 else 0.0
    if changed:
        current_video_index = 0
        load_next_video()
        logger.info("State -> %s (hold=%.1fs sticky=%s)", new_state, seconds, state_sticky)


def _expire_state_if_needed():
    global current_state, current_video_index, state_sticky
    if current_state == "idle" or state_sticky:
        return
    if time.time() < state_hold_until:
        return
    nxt = STATE_META.get(current_state, {}).get("next", "idle")
    if nxt == current_state:
        return
    logger.info("State '%s' expired -> '%s'", current_state, nxt)
    current_state = nxt if nxt in ANIMATION_CLASSES else "idle"
    current_video_index = 0
    state_sticky = False
    load_next_video()


def get_avatar_frame():
    """Advance the single VideoCapture owned by the avatar thread."""
    global last_surface, last_frame_np, cap_current, pending_reload, last_src_wh
    try:
        with video_lock:
            _apply_pending_state()
            if pending_reload:
                pending_reload = False
                load_next_video()
            _expire_state_if_needed()
            if cap_current is None or not cap_current.isOpened():
                load_next_video()

            ret, frame = (False, None)
            if cap_current is not None:
                ret, frame = cap_current.read()
                if not ret:
                    _expire_state_if_needed()
                    load_next_video()
                    if cap_current is not None:
                        ret, frame = cap_current.read()

            if not ret or frame is None:
                last_surface = fallback_surface or _blank_surface()
                return last_surface

            fh, fw = frame.shape[:2]
            if (fw, fh) != last_src_wh:
                last_src_wh = (fw, fh)
                resize_hud_to_frame(fw, fh)
            fitted = _fit_frame(frame, AVATAR_W, AVATAR_H)
            rgb = cv2.cvtColor(fitted, cv2.COLOR_BGR2RGB)
            last_frame_np = rgb
            last_surface = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
            return last_surface
    except Exception as e:
        logger.error("Avatar frame failed: %s", e)
        last_surface = fallback_surface or _blank_surface()
        return last_surface


def set_state(new_state: str, hold: Optional[float] = None):
    """Request a being-state change. The avatar thread applies it (thread-safe)."""
    global pending_state, pending_hold
    pending_state = new_state
    pending_hold = hold
    logger.info("State requested: %s", new_state)


def request_shutdown(reason: str = "keyboard"):
    global shutdown_flag
    if shutdown_flag:
        return
    logger.info("Shutdown requested (%s)", reason)
    shutdown_flag = True


def hotkey_listener():
    """Global Ctrl+Alt+Q / Ctrl+Alt+Esc — works even when the HUD is not focused."""
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    MOD_ALT, MOD_CONTROL, MOD_NOREPEAT = 0x0001, 0x0002, 0x4000
    WM_HOTKEY = 0x0312
    VK_Q, VK_ESCAPE, VK_G, VK_A = 0x51, 0x1B, 0x47, 0x41
    mods = MOD_CONTROL | MOD_ALT | MOD_NOREPEAT
    registered = []
    keymap = {
        1: ("Ctrl+Alt+Q", "quit"),
        2: ("Ctrl+Alt+Esc", "quit"),
        3: ("Ctrl+Alt+G", "gpu"),
        4: ("Ctrl+Alt+A", "ask"),
    }
    for hot_id, vk in ((1, VK_Q), (2, VK_ESCAPE), (3, VK_G), (4, VK_A)):
        if user32.RegisterHotKey(None, hot_id, mods, vk):
            registered.append(hot_id)
            logger.info("Hotkey armed: %s", keymap[hot_id][0])
        else:
            logger.warning("Could not register hotkey %s", keymap[hot_id][0])
    if not registered:
        return
    msg = wintypes.MSG()
    try:
        while not shutdown_flag:
            if user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):
                if msg.message == WM_HOTKEY:
                    action = keymap.get(int(msg.wParam), (None, None))[1]
                    if action == "quit":
                        request_shutdown("Ctrl+Alt+Q")
                        break
                    if action == "gpu":
                        Thread(target=dispatch_voice, args=("gpu status",), daemon=True).start()
                    elif action == "ask":
                        Thread(target=dispatch_voice, args=("what is an RTX GPU",), daemon=True).start()
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))
            else:
                time.sleep(0.05)
    finally:
        for hot_id in registered:
            user32.UnregisterHotKey(None, hot_id)


def avatar_loop():
    global last_surface, hud_user_placed
    clock = pygame.time.Clock()
    dragging = False
    drag_off = (0, 0)
    try:
        while not shutdown_flag:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # Frameless HUD plus Tk panels can emit a fake QUIT on Windows.
                    if not getattr(avatar_loop, "_ignored_quit", False):
                        logger.info("Ignoring pygame QUIT on frameless HUD — use Esc/Q or Ctrl+Alt+Q")
                        avatar_loop._ignored_quit = True
                    continue
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    request_shutdown("Esc/Q")
                    return
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    dragging = True
                    drag_off = event.pos
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    dragging = False
                    hud_user_placed = True
                elif event.type == pygame.MOUSEMOTION and dragging:
                    try:
                        import ctypes

                        class POINT(ctypes.Structure):
                            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

                        pt = POINT()
                        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
                        place_hud(pt.x - drag_off[0], pt.y - drag_off[1])
                        hud_user_placed = True
                    except Exception:
                        pass
            if screen is None:
                time.sleep(0.05)
                continue
            frame = get_avatar_frame()
            screen.fill((0, 0, 0))
            screen.blit(frame, (0, 0))
            pygame.display.flip()
            clock.tick(30)
    except Exception as e:
        logger.error("Avatar loop crashed: %s", e)
    finally:
        with video_lock:
            if cap_current is not None:
                try:
                    cap_current.release()
                except Exception:
                    pass


def cooler_mirror_loop():
    """Overlay GPU stats on a numpy snapshot. Does not touch pygame or VideoCapture."""
    out_path = os.path.join(_plugin_dir, "cooler_sylph.png")
    font_path = r"C:\Windows\Fonts\arialbd.ttf"
    while not shutdown_flag:
        try:
            with video_lock:
                raw = None if last_frame_np is None else last_frame_np.copy()
            if raw is None:
                time.sleep(2)
                continue
            img = Image.fromarray(raw).convert("RGBA")
            draw = ImageDraw.Draw(img)
            font = (
                ImageFont.truetype(font_path, 22)
                if os.path.isfile(font_path)
                else ImageFont.load_default()
            )
            gpu = None
            try:
                gpus = GPUtil.getGPUs()
                gpu = gpus[0] if gpus else None
            except Exception:
                gpu = None
            if gpu:
                stats = [
                    f"RTX {gpu.name.split()[-1]}",
                    f"Load: {gpu.load * 100:.0f}%",
                    f"Temp: {gpu.temperature}°C",
                ]
                try:
                    occ = _gpu_occupancy()
                    if occ:
                        extra = occ.overlay_lines()
                        stats.extend(extra[:3])
                except Exception:
                    pass
            else:
                stats = ["GPU unavailable"]
            for i, text in enumerate(stats):
                y = 10 + i * 28
                draw.text((12, y), text, fill="black", font=font)
                hot = gpu is not None and "°C" in text and gpu.temperature > 80
                draw.text((10, y), text, fill="red" if hot else "lime", font=font)
            draw.text((10, img.height - 28), "SYLPH Supreme", fill="white", font=font)
            img.save(out_path)
        except Exception as e:
            logger.error("Cooler mirror error: %s", e)
        for _ in range(30):
            if shutdown_flag:
                return
            time.sleep(1)


def init_avatar(run_loop_in_thread: bool = True):
    global screen, ANIMATION_CLASSES, AVATAR_W, AVATAR_H, COUNCIL_PX, _panels_thread
    ANIMATION_CLASSES = build_animation_classes()
    size_hud_for_library([path for pack in ANIMATION_CLASSES.values() for path in pack])
    COUNCIL_PX = min(340, inches_px(3.2))
    cx, cy = hud_center_pos()
    os.environ["SDL_VIDEO_WINDOW_POS"] = f"{cx},{cy}"
    logger.info(
        "HUD start %dx%d; council %dx%d (~3.2in); centered at %s,%s",
        AVATAR_W, AVATAR_H, COUNCIL_PX, COUNCIL_PX, cx, cy,
    )
    pygame.init()
    pygame.display.set_caption("RTX SYLPH v8.4")
    screen = pygame.display.set_mode((AVATAR_W, AVATAR_H), pygame.NOFRAME)
    pygame.mouse.set_visible(True)
    place_hud(cx, cy)
    load_fallback_surface()
    if run_loop_in_thread:
        Thread(target=avatar_loop, daemon=True, name="sylph-avatar").start()
    Thread(target=cooler_mirror_loop, daemon=True, name="sylph-cooler").start()
    if CLOCK_ENABLED or CHAT_ENABLED:
        _panels_thread = Thread(target=desktop_panels_loop, daemon=False, name="sylph-panels")
        _panels_thread.start()


# ---------------------------------------------------------------------------
# Wake word
# ---------------------------------------------------------------------------
recognizer = sr.Recognizer()
microphone = None


def resolve_mic_index(wanted: str) -> Optional[int]:
    """Pick a PyAudio capture device by name. Prefers a real mic over virtual mixers."""
    if not wanted:
        return None
    wanted = wanted.strip()
    if wanted.isdigit():
        return int(wanted)
    try:
        names = sr.Microphone.list_microphone_names() or []
    except Exception as e:
        logger.warning("Could not list microphones: %s", e)
        return None
    needle = wanted.lower()
    ranked = []
    for i, name in enumerate(names):
        if not name or needle not in name.lower():
            continue
        low = name.lower()
        score = 100 - i
        if low.startswith("microphone"):
            score += 30
        if "virtual" in low or "sonar" in low or "steam" in low or "mapper" in low:
            score -= 80
        ranked.append((score, i, name))
    if not ranked:
        logger.warning("No microphone matched '%s'", wanted)
        return None
    ranked.sort(reverse=True)
    score, index, name = ranked[0]
    logger.info("Microphone match: [%s] %s", index, name)
    return index


def init_microphone():
    global microphone
    try:
        index = resolve_mic_index(MIC_DEVICE)
        if index is None:
            microphone = sr.Microphone()
            logger.info("Microphone opened (Windows default) — wake word '%s'", WAKE_WORD)
        else:
            microphone = sr.Microphone(device_index=index)
            names = sr.Microphone.list_microphone_names()
            label = names[index] if names and index < len(names) else str(index)
            logger.info("Microphone opened: %s — wake word '%s'", label, WAKE_WORD)
        return True
    except Exception as e:
        logger.warning("Microphone unavailable (wake word disabled): %s", e)
        microphone = None
        return False


def _recognize(audio) -> str:
    try:
        return recognizer.recognize_google(audio).lower().strip()
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        logger.warning("Speech recognition request failed: %s", e)
        return ""


def _is_wake_word(token: str) -> bool:
    t = (token or "").strip(" .,!?").lower()
    if not t:
        return False
    if t == WAKE_WORD or t in WAKE_ALIASES:
        return True
    if t.startswith(WAKE_WORD) or WAKE_WORD.startswith(t) and len(t) >= 4:
        return True
    import difflib
    return difflib.SequenceMatcher(None, t, WAKE_WORD).ratio() >= 0.72


def _utterance_has_wake(text: str) -> bool:
    return any(_is_wake_word(w) for w in re.findall(r"[a-z']+", (text or "").lower()))


def _strip_wake(text: str) -> str:
    words = re.findall(r"[A-Za-z']+", text or "")
    kept = [w for w in words if not _is_wake_word(w)]
    cleaned = " ".join(kept)
    cleaned = re.sub(r"\b(hey|ok|okay|hi|hello|please|computer)\b", " ", cleaned, flags=re.I)
    return re.sub(r"\s+", " ", cleaned).strip(" .,!?")


def dispatch_voice(text: str):
    """Route a spoken command. Never called with the microphone lock held."""
    t = (text or "").strip()
    if not t:
        speak("I didn't catch that.")
        return
    logger.info("Dispatch: %s", t)
    try:
        if any(
            k in t
            for k in (
                "gpu",
                "graphics card",
                "how hot",
                "vram",
                "what's loaded",
                "whats loaded",
                "what is loaded",
                "who's on the gpu",
                "whos on the gpu",
                "occupancy",
            )
        ):
            gpu_status()
            return
        if any(k in t for k in ("system status", "cpu", "ram", "memory", "how am i running", "pc status")):
            sys_status()
            return
        if "volume" in t or "mute" in t or "unmute" in t:
            if "mute" in t and "unmute" not in t:
                volume_control("mute")
            elif "unmute" in t or "unmute" in t:
                volume_control("unmute")
            else:
                m = re.search(r"(\d{1,3})", t)
                volume_control("set", m.group(1) if m else None)
            return
        if "screenshot" in t or "screen shot" in t or "capture the screen" in t:
            take_screenshot()
            return
        if "lock" in t and any(k in t for k in ("pc", "computer", "workstation", "machine")):
            lock_pc()
            return
        if "close" in t and any(k in t for k in ("window", "windows", "council", "all")):
            close_all()
            return
        if "light" in t:
            action = "off" if "off" in t else "on" if "on" in t else "status"
            color = None
            for name in ("red", "green", "blue", "purple", "cyan", "white"):
                if name in t:
                    action, color = "color", name
                    break
            lights_control(action=action, color=color)
            return
        if "thermostat" in t or "temperature" in t:
            thermostat_control()
            return
        if "camera" in t or "doorbell" in t:
            if "spiral" in t or "all" in t:
                camera_spiral()
            else:
                camera_view()
            return
        if "airtag" in t or "find my" in t or "find my keys" in t:
            find_airtag()
            return
        if "screencast" in t or "cast" in t or "mirror" in t:
            screencast("start")
            return
        if any(k in t for k in ("what time", "world clock", "time zone", "time in", "clock")):
            speak(speakable_world_clock())
            return
        stream = streaming_intent(t)
        if stream:
            handle_streaming(stream)
            return
        desk = sylph_desk.handle_desk_intent(t)
        if desk == "SHOT":
            screenshot_ask()
            return
        if desk == "GOOGLE_HOME":
            launch_site("https://home.google.com")
            speak("Opening Google Home.")
            return
        if desk == "ALEXA_HOME":
            launch_site("https://alexa.amazon.com")
            speak("Opening Alexa.")
            return
        if desk == "GROK_BOT":
            launch_site(GROK_BOT_URL)
            speak("Opening Grok Bot.")
            return
        if desk == "GROK_MAIL":
            launch_site(GROK_BOT_URL)
            speak("Mail brief is on the clipboard. Paste it into Grok Bot. Drafts only.")
            return
        if isinstance(desk, str) and desk.startswith("CHROME"):
            open_chrome(desk[6:].strip())
            return
        if isinstance(desk, str) and desk.startswith("GROK_JOB "):
            launch_site(GROK_BOT_URL)
            job = desk.split(" ", 1)[-1].strip() or "job"
            speak(f"{job} brief is on the clipboard. Paste it into Grok Bot.")
            return
        if desk:
            speak(desk)
            chat_note("SYLPH", desk)
            return
        ask_ai(clean_spoken_question(t))
    except Exception as e:
        logger.error("Voice dispatch failed: %s", e)
        speak("That command failed.")


def wake_listener():
    if microphone is None:
        return
    try:
        recognizer.dynamic_energy_threshold = True
        recognizer.dynamic_energy_adjustment_damping = 0.15
        recognizer.dynamic_energy_ratio = 1.5
        recognizer.energy_threshold = 280
        # Wait through conversational pauses instead of clipping mid-sentence.
        recognizer.pause_threshold = 1.45
        recognizer.non_speaking_duration = 0.75
        recognizer.phrase_threshold = 0.3
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=1.1)
        logger.info(
            "Wake listener ready — say '%s' (energy=%s pause=%.2fs)",
            WAKE_WORD,
            int(recognizer.energy_threshold),
            recognizer.pause_threshold,
        )
    except Exception as e:
        logger.warning("Ambient noise calibration failed: %s", e)
    while not shutdown_flag:
        try:
            with microphone as source:
                audio = recognizer.listen(source, timeout=1.6, phrase_time_limit=14)
            text = _recognize(audio)
            if not text:
                continue
            logger.info("Heard: %s", text)
            if not _utterance_has_wake(text) and current_state != "listening" and pending_state != "listening":
                continue
            rest = _strip_wake(text)
            if sylph_desk.is_quiet() and not sylph_desk.is_quiet_wake(text) and not sylph_desk.is_quiet_wake(rest):
                logger.info("Quiet hours — ignored: %s", text)
                continue
            set_state("listening")
            if not rest:
                speak("Yes?")
                try:
                    with microphone as source:
                        follow = recognizer.listen(source, timeout=10, phrase_time_limit=20)
                    rest = _recognize(follow)
                    logger.info("Follow-up: %s", rest)
                except sr.WaitTimeoutError:
                    speak("Still here when you need me.")
                    continue
                except Exception as e:
                    logger.error("Follow-up listen failed: %s", e)
                    continue
            if rest:
                Thread(target=dispatch_voice, args=(rest,), daemon=True, name="sylph-dispatch").start()
        except sr.WaitTimeoutError:
            continue
        except Exception as e:
            logger.error("Wake error: %s", e)
            time.sleep(0.5)


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------
COUNCIL_DIR = os.path.join(_plugin_dir, "council_windows")


def clean_spoken_question(text: str) -> str:
    """Pull the real question out of 'open grok and ask ...' style speech."""
    t = text or ""
    t = re.sub(
        r"\b(can you|could you|would you|please|i want you to|i need you to)\b",
        " ",
        t,
        flags=re.I,
    )
    t = re.sub(
        r"\b(open|ask|query|call|use|launch|start)\s+"
        r"(grok|chatgpt|gemini(\s+(flash|pro))?|mistral|nemotron(\s+(lightning|super|ultra))?|lightning|llama|deepinfra|nvidia|the\s+council|ai\s+council|windows?)\b",
        " ",
        t,
        flags=re.I,
    )
    t = re.sub(r"\b(and ask|and then|and)\b", " ", t, flags=re.I)
    cleaned = re.sub(r"\s+", " ", t).strip(" .,!?")
    return cleaned or (text or "").strip()


def write_council_html(title: str, body: str, bg: str, fg: str, phase: str, question: str = "") -> str:
    os.makedirs(COUNCIL_DIR, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", title).strip("_")[:80] or "council"
    path = os.path.join(COUNCIL_DIR, f"{phase}_{slug}.html")
    qblock = (
        f"<div class='q'>Q: {html_lib.escape(question)}</div>" if question else ""
    )
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{html_lib.escape(title)}</title>"
        "<style>"
        "html,body{margin:0;padding:0;height:100%;box-sizing:border-box;}"
        f"body{{font-family:Consolas,'Cascadia Mono',monospace;background:{bg};color:{fg};padding:12px;overflow:auto;}}"
        "h1{font-size:13px;letter-spacing:1px;margin:0 0 8px;text-transform:uppercase;}"
        ".q{color:#ffe066;font-size:12px;margin:0 0 10px;line-height:1.35;}"
        "pre{white-space:pre-wrap;word-wrap:break-word;font-size:12px;line-height:1.4;margin:0;}"
        "</style></head><body>"
        f"<h1>{html_lib.escape(title)}</h1>"
        f"{qblock}"
        f"<pre>{html_lib.escape(body or '')}</pre>"
        "</body></html>"
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(html)
    return path


def spawn_window(url_or_path: str, position: Tuple[int, int], title: str = ""):
    try:
        gpus = GPUtil.getGPUs()
        if gpus and gpus[0].load > 0.9:
            logger.info("GPU high load — delaying window %s", title)
            time.sleep(1.5)
    except Exception:
        pass

    if os.path.isfile(url_or_path):
        url = Path(url_or_path).resolve().as_uri()
    else:
        url = url_or_path

    x, y = int(position[0]), int(position[1])
    side = max(260, min(340, int(COUNCIL_PX)))
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", title).strip("_")[:40] or "win"
    profile = os.path.join(COUNCIL_DIR, "profiles", slug)
    os.makedirs(profile, exist_ok=True)
    args = [
        BROWSER_PATH,
        f"--user-data-dir={profile}",
        "--no-first-run",
        "--no-default-browser-check",
        "--force-device-scale-factor=1",
        f"--window-size={side},{side}",
        f"--window-position={x},{y}",
        f"--app={url}",
    ]
    try:
        proc = subprocess.Popen(args)
        _spawned_procs.append(proc)
        logger.info("Window spawned: %s at (%s,%s) size=%s pid=%s", title, x, y, side, proc.pid)
        Thread(
            target=_force_window_rect,
            args=(proc.pid, x, y, side, side),
            daemon=True,
            name="sylph-place-win",
        ).start()
        return
    except Exception as e:
        logger.warning("Chrome-style spawn failed (%s): %s", title, e)
    try:
        if sys.platform == "win32":
            os.startfile(url_or_path if os.path.isfile(url_or_path) else url)
        else:
            subprocess.Popen([BROWSER_PATH, url])
        logger.info("Window spawned via fallback: %s", title)
    except Exception as e:
        logger.error("Window spawn failed: %s", e)


def screen_size() -> Tuple[int, int]:
    try:
        primary = screeninfo.get_monitors()[0]
        return primary.width, primary.height
    except Exception:
        return 1920, 1080


def _force_window_rect(pid: int, x: int, y: int, w: int, h: int, tries: int = 12):
    """Chrome often ignores --window-size. Pin the HWND after it exists."""
    if sys.platform != "win32" or not pid:
        return
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def _enum(hwnd, _lp):
        proc = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(proc))
        if proc.value == pid and user32.IsWindowVisible(hwnd):
            found.append(hwnd)
        return True

    for _ in range(tries):
        time.sleep(0.25)
        found.clear()
        user32.EnumWindows(_enum, 0)
        if not found:
            continue
        HWND_TOP = 0
        SWP_SHOWWINDOW = 0x0040
        for hwnd in found:
            user32.SetWindowPos(hwnd, HWND_TOP, int(x), int(y), int(w), int(h), SWP_SHOWWINDOW)
        return


def _council_positions() -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Place answer windows on two rings around the centered HUD. No overlap."""
    sw, sh = screen_size()
    size = max(260, min(340, int(COUNCIL_PX)))
    if hud_user_placed or hud_x or hud_y:
        cx = hud_x + AVATAR_W // 2
        cy = hud_y + AVATAR_H // 2
    else:
        cx, cy = sw // 2, sh // 2
    hw, hh = AVATAR_W / 2, AVATAR_H / 2
    gap = 36

    def radius(ang: float, extra: float = 0.0) -> float:
        c, s = abs(math.cos(ang)), abs(math.sin(ang))
        if c < 1e-6:
            edge = hh
        elif s < 1e-6:
            edge = hw
        else:
            edge = min(hw / c, hh / s)
        return edge + size / 2 + gap + extra

    def ring(n: int, extra: float, phase: float):
        pts = []
        for i in range(n):
            ang = -math.pi / 2 + phase + i * (2 * math.pi / n)
            r = radius(ang, extra)
            x = int(cx + r * math.cos(ang) - size / 2)
            y = int(cy + r * math.sin(ang) - size / 2)
            x = max(4, min(sw - size - 4, x))
            y = max(4, min(sh - size - 48, y))
            pts.append((x, y))
        return pts

    n = max(1, len(AI_PROVIDERS))
    return ring(n, 0.0, 0.0), ring(n, size + 20, math.pi / n)


# ---------------------------------------------------------------------------
# World clock — always-on frameless panel, 10 zones
# ---------------------------------------------------------------------------
CLOCK_ZONES = [
    ("Pacific", "America/Los_Angeles", -8),
    ("Eastern", "America/New_York", -5),
    ("London", "Europe/London", 0),
    ("Berlin", "Europe/Berlin", 1),
    ("UTC", "UTC", 0),
    ("India", "Asia/Kolkata", 5.5),
    ("Beijing", "Asia/Shanghai", 8),
    ("Tokyo", "Asia/Tokyo", 9),
    ("Sydney", "Australia/Sydney", 10),
    ("São Paulo", "America/Sao_Paulo", -3),
]


def _zone_now(iana: str, fallback_hours: float) -> datetime:
    if ZoneInfo is not None:
        try:
            return datetime.now(ZoneInfo(iana))
        except Exception:
            pass
    offset = timedelta(hours=fallback_hours)
    return datetime.now(timezone(offset))


def world_clock_rows() -> List[Tuple[str, str, str]]:
    rows = []
    for label, iana, offset in CLOCK_ZONES:
        now = _zone_now(iana, offset)
        tz = now.tzname() or label
        rows.append((label, now.strftime("%H:%M:%S"), tz))
    return rows


def speakable_world_clock() -> str:
    bits = []
    for label, hhmmss, tz in world_clock_rows():
        bits.append(f"{label} {hhmmss[:-3]} {tz}")
    return "World clock: " + ". ".join(bits[:6]) + "."


# ---------------------------------------------------------------------------
# Watch desk — launch Netflix / Prime; suggest via TMDB (legal catalog) + Grok
# Does not scrape Netflix or Amazon. Opens the user's own browser/app.
# ---------------------------------------------------------------------------
NETFLIX_HOME = "https://www.netflix.com/browse"
NETFLIX_NEW = "https://www.netflix.com/latest"
PRIME_HOME = "https://www.primevideo.com"
PRIME_SEARCH = "https://www.primevideo.com/search/ref=atv_nb_sr?phrase="
NETFLIX_SEARCH = "https://www.netflix.com/search?q="
TMDB_NETFLIX_ID = 8
TMDB_PRIME_ID = 9
_watch_cache: Dict[str, object] = {"ts": 0.0, "netflix": [], "prime": [], "source": "", "blurb": ""}
_watch_ui = {"lines": None, "source": None, "refresh": None}


def open_chrome(url: str = "") -> str:
    """Open a real Chrome window (not the council --app ring)."""
    raw = (url or "").strip()
    if not raw or raw.lower() in ("chrome", "a window", "new window", "new tab", "browser"):
        target = "chrome://newtab"
    elif re.match(r"^https?://", raw, re.I) or raw.lower().startswith("chrome://"):
        target = raw
    elif re.match(r"^[\w.-]+\.[a-z]{2,}(/[\S]*)?$", raw, re.I):
        target = "https://" + raw.lstrip("/")
    else:
        target = "https://www.google.com/search?q=" + quote(raw)
    args = [BROWSER_PATH, "--new-window", target]
    try:
        proc = subprocess.Popen(args)
        _spawned_procs.append(proc)
        logger.info("Chrome window: %s pid=%s", target, proc.pid)
        speak("Opening Chrome.")
        chat_note("SYS", f"Opened Chrome: {target}")
        return f"Opened Chrome: {target}"
    except Exception as e:
        logger.warning("Chrome launch failed: %s", e)
        if launch_site(target):
            speak("Opened the page.")
            return f"Opened {target}"
        return f"Could not open Chrome: {e}"


def launch_site(url: str) -> bool:
    try:
        if sys.platform == "win32":
            os.startfile(url)
        else:
            subprocess.Popen([BROWSER_PATH, url])
        return True
    except Exception as e:
        logger.warning("Launch failed %s: %s", url, e)
        return False


def launch_streaming_app(kind: str) -> str:
    kind = (kind or "").lower()
    if kind in ("netflix", "nflix"):
        for target in ("netflix:", NETFLIX_HOME):
            if launch_site(target):
                set_state("sound_system")
                speak("Opening Netflix.")
                chat_note("SYS", "Opened Netflix in your browser or app.")
                return "Opened Netflix"
        return "Could not open Netflix"
    for target in ("primevideo:", "amazonvideo:", PRIME_HOME):
        if launch_site(target):
            set_state("sound_system")
            speak("Opening Prime Video.")
            chat_note("SYS", "Opened Prime Video in your browser or app.")
            return "Opened Prime Video"
    return "Could not open Prime Video"


def _tmdb_headers() -> dict:
    headers = {"Accept": "application/json"}
    if TMDB_READ_TOKEN:
        headers["Authorization"] = f"Bearer {TMDB_READ_TOKEN}"
    return headers


def _watch_item_label(item) -> str:
    if isinstance(item, dict):
        title = item.get("title") or ""
        year = item.get("year") or ""
        return f"{title} ({year})" if year else title
    return str(item)


def _watch_item_overview(item) -> str:
    if isinstance(item, dict):
        return (item.get("overview") or "").strip()
    return ""


def _tmdb_discover(provider_id: int, media: str) -> list:
    if not TMDB_API_KEY and not TMDB_READ_TOKEN:
        return []
    try:
        params = {
            "watch_region": WATCH_REGION,
            "with_watch_providers": str(provider_id),
            "with_watch_monetization_types": "flatrate",
            "sort_by": "popularity.desc",
            "language": "en-US",
            "page": 1,
        }
        if TMDB_API_KEY:
            params["api_key"] = TMDB_API_KEY
        resp = requests.get(
            f"https://api.themoviedb.org/3/discover/{media}",
            params=params,
            headers=_tmdb_headers(),
            timeout=12,
        )
        if resp.status_code >= 400:
            logger.warning("TMDB HTTP %s: %s", resp.status_code, (resp.text or "")[:200])
            return []
        rows = []
        for item in (resp.json().get("results") or [])[:8]:
            title = (item.get("title") or item.get("name") or "").strip()
            year = (item.get("release_date") or item.get("first_air_date") or "")[:4]
            overview = (item.get("overview") or "").strip()
            if title:
                rows.append({"title": title, "year": year, "overview": overview, "media": media})
        return rows
    except Exception as e:
        logger.warning("TMDB failed: %s", e)
        return []


def _tmdb_catalog(provider_id: int) -> list:
    seen = set()
    out = []
    for media in ("tv", "movie"):
        for item in _tmdb_discover(provider_id, media):
            key = item["title"].lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
            if len(out) >= 5:
                return out
    return out


def _grok_watch_picks(platform: str) -> list:
    if not GROK_API_KEY:
        return []
    prompt = (
        f"Recommend 4 notable titles currently discussed as being on {platform} in the {WATCH_REGION}. "
        "For each title use this exact format:\n"
        "TITLE | YEAR | FULL SYNOPSIS IN 2-4 SENTENCES\n"
        "No numbering, no URLs, no extra intro."
    )
    raw = query_grok_chat(prompt, None)
    rows = []
    for line in (raw or "").splitlines():
        line = re.sub(r"^[\s\-\*\d\.]+", "", line).strip()
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 2:
            continue
        title, year = parts[0], parts[1][:4]
        overview = " ".join(parts[2:]).strip() if len(parts) > 2 else ""
        if title:
            rows.append({"title": title, "year": year, "overview": overview, "media": "mixed"})
        if len(rows) >= 4:
            break
    return rows


def refresh_watch_desk(force: bool = False) -> Dict[str, object]:
    now = time.time()
    if not force and _watch_cache["ts"] and now - float(_watch_cache["ts"]) < 1800:
        return _watch_cache
    nflix = _tmdb_catalog(TMDB_NETFLIX_ID)
    prime = _tmdb_catalog(TMDB_PRIME_ID)
    source = "TMDB"
    if not nflix:
        nflix = _grok_watch_picks("Netflix")
        source = "Grok"
    if not prime:
        prime = _grok_watch_picks("Amazon Prime Video")
        source = "TMDB + Grok" if source == "TMDB" else "Grok"
    _watch_cache.update(
        {
            "ts": now,
            "netflix": nflix[:5],
            "prime": prime[:5],
            "source": source,
            "blurb": TMDB_ATTRIBUTION + " Not affiliated with Netflix or Amazon.",
        }
    )
    return _watch_cache


def speakable_watch(platform: str = "both") -> str:
    data = refresh_watch_desk()
    chunks = []

    def block(name: str, items):
        if not items:
            return
        chunks.append(f"On {name}.")
        for item in items[:4]:
            label = _watch_item_label(item)
            overview = _watch_item_overview(item)
            if overview:
                chunks.append(f"{label}. {overview}")
            else:
                chunks.append(label + ".")

    if platform in ("both", "netflix"):
        block("Netflix", data.get("netflix") or [])
    if platform in ("both", "prime"):
        block("Prime Video", data.get("prime") or [])
    if not chunks:
        return "I can open Netflix or Prime in your browser. Catalog suggestions need TMDB or Grok."
    chunks.append(TMDB_ATTRIBUTION)
    return " ".join(chunks)


def streaming_intent(t: str) -> Optional[str]:
    low = (t or "").lower()
    netflix = "netflix" in low
    prime = any(k in low for k in ("prime video", "amazon prime", "prime video", "amazon video"))
    if "prime" in low and any(k in low for k in ("watch", "video", "amazon", "stream", "show", "movie", "open", "launch")):
        prime = True
    open_it = any(k in low for k in ("open", "launch", "start", "put on", "fire up", "pull up"))
    suggest = any(
        k in low
        for k in (
            "what's new", "whats new", "what is new", "what to watch",
            "should i watch", "recommend", "suggest", "what's on", "whats on",
            "new on", "tonight", "something to watch",
        )
    )
    if open_it and netflix:
        return "open_netflix"
    if open_it and prime:
        return "open_prime"
    if suggest and netflix and not prime:
        return "suggest_netflix"
    if suggest and prime and not netflix:
        return "suggest_prime"
    if suggest and (netflix or prime or "watch" in low or "stream" in low):
        return "suggest_both"
    if netflix and not prime and any(k in low for k in ("new", "watch", "show", "movie", "series")):
        return "suggest_netflix"
    if prime and not netflix:
        return "suggest_prime"
    if low.strip() in ("netflix", "open netflix"):
        return "open_netflix"
    if low.strip() in ("prime", "prime video", "amazon prime"):
        return "open_prime"
    return None


def handle_streaming(intent: str) -> str:
    if intent == "open_netflix":
        return launch_streaming_app("netflix")
    if intent == "open_prime":
        return launch_streaming_app("prime")
    platform = "netflix" if intent == "suggest_netflix" else "prime" if intent == "suggest_prime" else "both"
    set_state("sound_system")
    refresh_watch_desk(force=True)
    text = speakable_watch(platform)
    speak(text)
    chat_note("SYLPH", text)
    if _watch_ui.get("refresh"):
        try:
            _watch_ui["refresh"]()
        except Exception:
            pass
    return text


def open_title_search(platform: str, title: str):
    q = re.sub(r"\s+\(\d{4}\)\s*$", "", title or "").strip()
    if not q:
        launch_streaming_app(platform)
        return
    if platform == "netflix":
        launch_site(NETFLIX_SEARCH + quote(q))
    else:
        launch_site(PRIME_SEARCH + quote(q))


INBOX_DIR = os.path.join(_plugin_dir, "council_windows", "inbox")
_chat_ui_q: Queue = Queue()
URL_RE = re.compile(r"https?://[^\s<>\"']+", re.I)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
TEXT_EXTS = {
    ".txt", ".md", ".json", ".csv", ".log", ".py", ".js", ".ts", ".html",
    ".css", ".xml", ".yml", ".yaml", ".toml", ".ini", ".bat", ".ps1", ".c",
    ".cpp", ".h", ".rs", ".go", ".java",
}


def chat_note(who: str, text: str):
    if not text:
        return
    try:
        _chat_ui_q.put_nowait((who, text))
    except Exception:
        pass


def _inbox_copy(src: str) -> str:
    os.makedirs(INBOX_DIR, exist_ok=True)
    name = os.path.basename(src)
    dest = os.path.join(INBOX_DIR, f"{int(time.time())}_{name}")
    try:
        from shutil import copy2

        copy2(src, dest)
        return dest
    except Exception:
        return src


def _image_to_b64(path: str, max_side: int = 1280) -> Optional[Tuple[str, str]]:
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=82)
        return "image/jpeg", base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as e:
        logger.warning("Could not encode image %s: %s", path, e)
        return None


def _video_poster(path: str) -> Optional[str]:
    cap = None
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 30.0
        target = min(max(frames // 4, 1), int(fps * 2))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok or frame is None:
            return None
        os.makedirs(INBOX_DIR, exist_ok=True)
        out = os.path.join(INBOX_DIR, f"{int(time.time())}_poster.jpg")
        cv2.imwrite(out, frame)
        return out
    except Exception as e:
        logger.warning("Video poster failed (%s): %s", path, e)
        return None
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass


def _read_text_file(path: str, limit: int = 12000) -> str:
    try:
        raw = Path(path).read_bytes()
        if b"\x00" in raw[:1024]:
            return f"(binary file, {len(raw)} bytes)"
        text = raw.decode("utf-8", errors="replace")
        if len(text) > limit:
            return text[:limit] + "\n…[truncated]"
        return text
    except Exception as e:
        return f"(could not read: {e})"


def _fetch_link(url: str) -> str:
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent": "RTX-SYLPH/8.0"})
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if resp.status_code >= 400:
            return f"{url} → HTTP {resp.status_code}"
        if "html" in ctype or url.lower().endswith((".html", ".htm")) or "<html" in (resp.text[:400].lower()):
            html = resp.text[:80000]
            title = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
            title_txt = re.sub(r"\s+", " ", title.group(1)).strip() if title else ""
            text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
            text = re.sub(r"(?s)<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text).strip()[:3000]
            return f"Title: {title_txt}\n{text}"
        return f"{url} ({ctype or 'unknown'}, {len(resp.content)} bytes)"
    except Exception as e:
        return f"{url} → fetch failed: {e}"


def compose_chat_payload(text: str, paths: List[str]) -> Tuple[str, List[Tuple[str, str]]]:
    chunks = []
    images: List[Tuple[str, str]] = []
    body = (text or "").strip()
    if body:
        chunks.append(body)
    for url in URL_RE.findall(body):
        excerpt = _fetch_link(url)
        chunks.append(f"[link {url}]\n{excerpt}")
        title = ""
        if excerpt.startswith("Title:"):
            title = excerpt.split("\n", 1)[0].replace("Title:", "").strip()
        sylph_desk.save_link(url, title)
    for raw in paths:
        path = raw
        if not os.path.isfile(path):
            chunks.append(f"[missing file: {raw}]")
            continue
        ext = os.path.splitext(path)[1].lower()
        size = os.path.getsize(path)
        if ext in IMAGE_EXTS:
            encoded = _image_to_b64(path)
            if encoded:
                images.append(encoded)
            chunks.append(f"[image {os.path.basename(path)} {size} bytes]")
        elif ext in VIDEO_EXTS:
            poster = _video_poster(path)
            if poster:
                encoded = _image_to_b64(poster)
                if encoded:
                    images.append(encoded)
            chunks.append(f"[video {os.path.basename(path)} {size} bytes — poster frame attached]")
        elif ext in TEXT_EXTS or size < 256000:
            chunks.append(f"[file {os.path.basename(path)}]\n{_read_text_file(path)}")
        else:
            chunks.append(f"[file {os.path.basename(path)} {size} bytes — not inlined]")
    prompt = "\n\n".join(chunks).strip() or "The user sent attachments with no caption."
    return prompt, images


def _sylph_chat_system() -> str:
    k = sylph_license.knobs()
    wit = k["wit"]
    if sylph_license.is_premium():
        flavor = "Keep the carbon-fiber one-liners" if wit >= 60 else "Keep jokes rare; prefer dry precision"
        return (
            "You are RTX SYLPH, Flight edition — holographic desktop crewmate running full xAI Grok. "
            f"{flavor}. Accuracy still wins. "
            "Never invent 'not released yet' for products that already shipped (RTX 5090, RTX PRO 6000 Blackwell, Grace Blackwell GB200). "
            "If unsure, say what you know and what to verify. Use attachments. "
            "Answer completely; do not truncate hardware facts to be cute."
        )
    return (
        "You are RTX SYLPH, studio edition — a holographic RTX desktop companion. "
        "Snappy, quirky, a little vague on bleeding-edge SKUs if you must choose vibe over a spec sheet. "
        "Use attachments. Be useful first. Short answers."
    )


def _gen_knobs() -> dict:
    k = sylph_license.knobs()
    return {"temperature": k["temperature"], "max_tokens": k["max_tokens"]}


def query_grok_chat(prompt: str, images: Optional[List[Tuple[str, str]]] = None) -> str:
    if not GROK_API_KEY:
        return "Grok API key missing in config.json"
    prompt = sylph_license.enrich_prompt(prompt)
    if images:
        content = [{"type": "text", "text": prompt}]
        for mime, b64 in images[:4]:
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
            )
        user_msg = {"role": "user", "content": content}
    else:
        user_msg = {"role": "user", "content": prompt}
    knobs = _gen_knobs()
    return _chat_completion(
        "https://api.x.ai/v1/chat/completions",
        {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"},
        {
            "model": GROK_MODEL,
            "messages": [
                {"role": "system", "content": _sylph_chat_system()},
                user_msg,
            ],
            "temperature": knobs["temperature"],
            "max_tokens": max(knobs["max_tokens"], 1200 if images else knobs["max_tokens"]),
        },
        "SYLPH",
        timeout=90 if sylph_license.is_premium() else 60,
    )


def handle_chat_message(text: str, paths: List[str], council: bool = False) -> str:
    if not paths and not council:
        stream = streaming_intent(text)
        if stream:
            return handle_streaming(stream)
        desk = sylph_desk.handle_desk_intent(text)
        if desk == "SHOT":
            return screenshot_ask()
        if desk == "GOOGLE_HOME":
            launch_site("https://home.google.com")
            return "Opened Google Home"
        if desk == "ALEXA_HOME":
            launch_site("https://alexa.amazon.com")
            return "Opened Alexa"
        if desk == "GROK_BOT":
            launch_site(GROK_BOT_URL)
            return "Opened Grok Bot"
        if desk == "GROK_MAIL":
            launch_site(GROK_BOT_URL)
            speak("Mail brief is on the clipboard. Paste it into Grok Bot. Drafts only.")
            return "Opened Grok Bot with MAIL brief on clipboard"
        if isinstance(desk, str) and desk.startswith("CHROME"):
            return open_chrome(desk[6:].strip())
        if isinstance(desk, str) and desk.startswith("GROK_JOB "):
            launch_site(GROK_BOT_URL)
            job = desk.split(" ", 1)[-1].strip() or "job"
            speak(f"{job} brief is on the clipboard. Paste it into Grok Bot.")
            return f"Opened Grok Bot with {job} brief on clipboard"
        if desk:
            chat_note("SYLPH", desk)
            speak(desk)
            return desk
    prompt, images = compose_chat_payload(text, paths)
    logger.info("Chat %s attachments=%s council=%s", prompt[:180].replace("\n", " "), len(paths), council)
    set_state("thinking")
    reply = query_grok_chat(prompt, images)
    chat_note("SYLPH", reply)
    if council:
        Thread(target=ask_ai, args=(prompt,), daemon=True, name="sylph-chat-council").start()
    else:
        spoken = _spoken_brief(reply, 220)
        if spoken and "API key missing" not in spoken and "error" not in spoken.lower()[:24]:
            speak(spoken)
        set_state("answering", hold=8)
    return reply


def _bind_drag(win, handle):
    drag = {"x": 0, "y": 0}

    def start_move(event):
        drag["x"], drag["y"] = event.x_root, event.y_root

    def on_move(event):
        dx = event.x_root - drag["x"]
        dy = event.y_root - drag["y"]
        drag["x"], drag["y"] = event.x_root, event.y_root
        win.geometry(f"+{win.winfo_x() + dx}+{win.winfo_y() + dy}")

    handle.bind("<Button-1>", start_move)
    handle.bind("<B1-Motion>", on_move)


def _build_clock_panel(tk, parent, sw, sh):
    clock = tk.Toplevel(parent)
    clock.overrideredirect(True)
    clock.attributes("-topmost", True)
    clock.configure(bg="#001114")
    width, height = 236, 412
    x = max(8, sw - width - 16)
    y = max(8, (sh - height) // 2)
    clock.geometry(f"{width}x{height}+{x}+{y}")
    title = tk.Label(
        clock,
        text="SYLPH  WORLD CLOCK",
        fg="#00ffd2",
        bg="#001114",
        font=("Consolas", 10, "bold"),
    )
    title.pack(pady=(10, 4))
    _bind_drag(clock, title)
    _bind_drag(clock, clock)
    tk.Frame(clock, bg="#00ffd2", height=1).pack(fill="x", padx=12, pady=(0, 6))
    labels = []
    for _ in CLOCK_ZONES:
        row = tk.Frame(clock, bg="#001114")
        row.pack(fill="x", padx=14, pady=2)
        city = tk.Label(row, text="", fg="#7af0ff", bg="#001114", font=("Consolas", 10), width=10, anchor="w")
        city.pack(side="left")
        time_lbl = tk.Label(row, text="", fg="#00ffd2", bg="#001114", font=("Consolas", 13, "bold"), width=9, anchor="e")
        time_lbl.pack(side="left")
        tz_lbl = tk.Label(row, text="", fg="#66aa99", bg="#001114", font=("Consolas", 9), width=5, anchor="e")
        tz_lbl.pack(side="right")
        labels.append((city, time_lbl, tz_lbl))
    logger.info("World clock armed at %sx%s", x, y)
    return clock, labels


def _build_chat_panel(tk, filedialog, parent, sw, sh):
    chat = tk.Toplevel(parent)
    chat.overrideredirect(True)
    chat.attributes("-topmost", True)
    chat.configure(bg="#001114")
    width, height = 420, 560
    x, y = 12, max(8, (sh - height) // 2)
    chat.geometry(f"{width}x{height}+{x}+{y}")

    header = tk.Frame(chat, bg="#001114")
    header.pack(fill="x", padx=10, pady=(8, 0))
    title = tk.Label(
        header,
        text="SYLPH  CONSOLE",
        fg="#00ffd2",
        bg="#001114",
        font=("Consolas", 11, "bold"),
        anchor="w",
    )
    title.pack(side="left")
    hint = tk.Label(
        header,
        text="text · files · jpg · mp4 · links",
        fg="#66aa99",
        bg="#001114",
        font=("Consolas", 8),
    )
    hint.pack(side="right")
    _bind_drag(chat, header)
    _bind_drag(chat, title)
    tk.Frame(chat, bg="#00ffd2", height=1).pack(fill="x", padx=10, pady=(6, 6))

    transcript = tk.Text(
        chat,
        bg="#00181c",
        fg="#d8fff4",
        insertbackground="#00ffd2",
        font=("Consolas", 10),
        wrap="word",
        relief="flat",
        padx=8,
        pady=8,
        height=18,
    )
    transcript.pack(fill="both", expand=True, padx=10)
    transcript.tag_config("YOU", foreground="#ffe066", font=("Consolas", 10, "bold"))
    transcript.tag_config("SYLPH", foreground="#00ffd2", font=("Consolas", 10, "bold"))
    transcript.tag_config("SYS", foreground="#66aa99")
    transcript.insert("end", "SYS\n", "SYS")
    transcript.insert(
        "end",
        "Type to me. Attach images, video, or files. Paste a URL. Ctrl+Enter sends. Council opens the ring.\n\n",
    )
    transcript.configure(state="disabled")

    attached: List[str] = []
    attach_var = tk.StringVar(value="no files attached")
    tk.Label(
        chat,
        textvariable=attach_var,
        fg="#7af0ff",
        bg="#001114",
        font=("Consolas", 8),
        wraplength=390,
        justify="left",
        anchor="w",
    ).pack(fill="x", padx=10, pady=(4, 0))

    entry = tk.Text(
        chat,
        bg="#002226",
        fg="#e8fff8",
        insertbackground="#00ffd2",
        font=("Consolas", 10),
        wrap="word",
        height=4,
        relief="flat",
        padx=8,
        pady=6,
    )
    entry.pack(fill="x", padx=10, pady=(6, 4))
    entry.focus_set()

    btns = tk.Frame(chat, bg="#001114")
    btns.pack(fill="x", padx=10, pady=(0, 10))
    busy = {"on": False}

    def refresh_attach():
        if not attached:
            attach_var.set("no files attached")
        else:
            names = ", ".join(os.path.basename(p) for p in attached)
            attach_var.set(f"attached: {names}")

    def append_line(who: str, text: str):
        transcript.configure(state="normal")
        transcript.insert("end", f"{who}\n", who if who in ("YOU", "SYLPH") else "SYS")
        transcript.insert("end", (text or "").strip() + "\n\n")
        transcript.see("end")
        transcript.configure(state="disabled")

    def add_paths(paths: List[str]):
        for path in paths:
            if path and os.path.isfile(path) and path not in attached:
                attached.append(_inbox_copy(path) if INBOX_DIR not in os.path.abspath(path) else path)
        refresh_attach()

    def do_attach():
        paths = filedialog.askopenfilenames(
            parent=chat,
            title="Attach for SYLPH",
            filetypes=[
                ("Media & files", "*.jpg *.jpeg *.png *.webp *.gif *.mp4 *.mov *.mkv *.txt *.md *.json *.py *.log"),
                ("Images", "*.jpg *.jpeg *.png *.webp *.gif *.bmp"),
                ("Video", "*.mp4 *.mov *.mkv *.webm *.avi"),
                ("All files", "*.*"),
            ],
        )
        add_paths(list(paths or []))

    def do_paste():
        try:
            from PIL import ImageGrab

            clip = ImageGrab.grabclipboard()
        except Exception:
            clip = None
        if isinstance(clip, Image.Image):
            os.makedirs(INBOX_DIR, exist_ok=True)
            path = os.path.join(INBOX_DIR, f"{int(time.time())}_paste.jpg")
            clip.convert("RGB").save(path, "JPEG", quality=88)
            add_paths([path])
            append_line("SYS", f"Pasted image → {os.path.basename(path)}")
            return
        if isinstance(clip, list):
            add_paths([p for p in clip if isinstance(p, str)])
            return
        try:
            clip_text = chat.clipboard_get()
        except Exception:
            clip_text = ""
        if clip_text:
            entry.insert("insert", clip_text)

    def do_send(council: bool = False):
        if busy["on"]:
            return
        text = entry.get("1.0", "end").strip()
        paths = list(attached)
        if not text and not paths:
            return
        busy["on"] = True
        shown = text or "(attachments only)"
        if paths:
            shown += "\n" + " ".join("📎 " + os.path.basename(p) for p in paths)
        append_line("YOU", shown)
        entry.delete("1.0", "end")
        attached.clear()
        refresh_attach()

        def worker():
            try:
                handle_chat_message(text, paths, council=council)
            except Exception as e:
                logger.error("Chat send failed: %s", e)
                chat_note("SYS", f"Send failed: {e}")
            finally:
                busy["on"] = False

        Thread(target=worker, daemon=True, name="sylph-chat-send").start()

    def poll_notes():
        if shutdown_flag:
            return
        try:
            while True:
                who, text = _chat_ui_q.get_nowait()
                append_line(who, text)
        except Empty:
            pass
        chat.after(200, poll_notes)

    def on_keys(event):
        if event.state & 0x4 and event.keysym.lower() in ("return", "kp_enter"):
            do_send(False)
            return "break"
        if event.keysym.lower() == "v" and event.state & 0x4:
            do_paste()
            return "break"
        return None

    entry.bind("<KeyPress>", on_keys)

    def mkbtn(label, cmd, side="left"):
        b = tk.Button(
            btns,
            text=label,
            command=cmd,
            bg="#003a3a",
            fg="#00ffd2",
            activebackground="#00ffd2",
            activeforeground="#001114",
            relief="flat",
            font=("Consolas", 9, "bold"),
            padx=8,
            pady=4,
        )
        b.pack(side=side, padx=3)
        return b

    mkbtn("Attach", do_attach)
    mkbtn("Paste", do_paste)
    mkbtn("Council", lambda: do_send(True), side="right")
    mkbtn("Send", lambda: do_send(False), side="right")
    poll_notes()
    logger.info("Chat console armed at %sx%s", x, y)
    return chat


def _build_watch_panel(tk, parent, sw, sh):
    watch = tk.Toplevel(parent)
    watch.overrideredirect(True)
    watch.attributes("-topmost", True)
    watch.configure(bg="#001114")
    width, height = 380, 340
    x = max(8, sw - width - 16)
    y = max(8, sh - height - 52)
    watch.geometry(f"{width}x{height}+{x}+{y}")
    header = tk.Frame(watch, bg="#001114")
    header.pack(fill="x", padx=10, pady=(8, 0))
    title = tk.Label(
        header,
        text="SYLPH  WATCH",
        fg="#00ffd2",
        bg="#001114",
        font=("Consolas", 10, "bold"),
        anchor="w",
    )
    title.pack(side="left")
    _bind_drag(watch, header)
    _bind_drag(watch, title)
    tk.Frame(watch, bg="#00ffd2", height=1).pack(fill="x", padx=10, pady=(6, 4))
    body = tk.Text(
        watch,
        bg="#00181c",
        fg="#d8fff4",
        font=("Consolas", 9),
        wrap="word",
        relief="flat",
        height=10,
        padx=8,
        pady=6,
        cursor="arrow",
    )
    body.pack(fill="both", expand=True, padx=10)
    body.tag_config("H", foreground="#ffe066", font=("Consolas", 9, "bold"))
    body.tag_config("T", foreground="#00ffd2", font=("Consolas", 9, "bold"))
    body.tag_config("O", foreground="#c8fff0")
    body.tag_config("F", foreground="#66aa99")
    src_var = tk.StringVar(value=TMDB_ATTRIBUTION)

    def paint():
        data = _watch_cache
        body.configure(state="normal")
        body.delete("1.0", "end")

        def dump(heading, items, empty):
            body.insert("end", heading + "\n", "H")
            rows = items or [empty]
            for item in rows[:4]:
                if isinstance(item, dict):
                    body.insert("end", _watch_item_label(item) + "\n", "T")
                    ov = _watch_item_overview(item)
                    if ov:
                        body.insert("end", ov + "\n\n", "O")
                    else:
                        body.insert("end", "\n")
                else:
                    body.insert("end", f"  {item}\n", "O")

        dump("NETFLIX", data.get("netflix"), "Say what's new on Netflix")
        dump("PRIME VIDEO", data.get("prime"), "Say what's new on Prime")
        body.insert("end", TMDB_ATTRIBUTION + "\nNot affiliated with Netflix or Amazon.\n", "F")
        body.configure(state="disabled")
        src_var.set(str(data.get("blurb") or TMDB_ATTRIBUTION))

    def do_refresh():
        def work():
            refresh_watch_desk(force=True)
            _watch_ui["dirty"] = True

        Thread(target=work, daemon=True, name="sylph-watch").start()

    _watch_ui["paint"] = paint
    _watch_ui["refresh"] = lambda: _watch_ui.__setitem__("dirty", True)
    attr = tk.Frame(watch, bg="#001114")
    attr.pack(fill="x", padx=10, pady=(2, 0))
    logo_path = os.path.join(_plugin_dir, "assets", "tmdb_logo.png")
    if os.path.isfile(logo_path):
        try:
            watch._tmdb_logo = tk.PhotoImage(file=logo_path)
            tk.Label(attr, image=watch._tmdb_logo, bg="#001114").pack(anchor="w")
        except Exception as e:
            logger.warning("TMDB logo not shown: %s", e)
    tk.Label(
        attr,
        textvariable=src_var,
        fg="#90cea1",
        bg="#001114",
        font=("Consolas", 7),
        wraplength=350,
        justify="left",
        anchor="w",
    ).pack(fill="x", pady=(2, 0))
    btns = tk.Frame(watch, bg="#001114")
    btns.pack(fill="x", padx=8, pady=(4, 8))

    def mk(label, cmd):
        tk.Button(
            btns,
            text=label,
            command=cmd,
            bg="#003a3a",
            fg="#00ffd2",
            activebackground="#00ffd2",
            activeforeground="#001114",
            relief="flat",
            font=("Consolas", 8, "bold"),
            padx=6,
            pady=3,
        ).pack(side="left", padx=3)

    mk("Netflix", lambda: launch_streaming_app("netflix"))
    mk("Prime", lambda: launch_streaming_app("prime"))
    mk("New", do_refresh)
    paint()
    watch.after(800, do_refresh)
    logger.info("Watch desk armed at %sx%s", x, y)
    return watch


def desktop_panels_loop():
    """One Tk thread owns the clock and the text console (avoids Tcl teardown crashes)."""
    global _clock_root
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as e:
        logger.warning("Desktop panels skipped (tkinter unavailable): %s", e)
        return
    try:
        root = tk.Tk()
        _clock_root = root
        root.withdraw()
        root.attributes("-topmost", True)
        sw, sh = screen_size()
        clock_labels = []
        if CLOCK_ENABLED:
            _clock, clock_labels = _build_clock_panel(tk, root, sw, sh)
        if CHAT_ENABLED:
            _build_chat_panel(tk, filedialog, root, sw, sh)
        if WATCH_ENABLED:
            _build_watch_panel(tk, root, sw, sh)
        if DOCK_ENABLED:
            sylph_desk.build_dock(tk, root, sw, sh, sys.modules[__name__])

        def tick():
            if shutdown_flag:
                try:
                    root.quit()
                    root.destroy()
                except Exception:
                    pass
                return
            if clock_labels:
                for (city, time_lbl, tz_lbl), (label, hhmmss, tz) in zip(clock_labels, world_clock_rows()):
                    city.config(text=label.upper())
                    time_lbl.config(text=hhmmss)
                    tz_lbl.config(text=(tz or "")[:5])
            if _watch_ui.get("dirty") and _watch_ui.get("paint"):
                _watch_ui["dirty"] = False
                try:
                    _watch_ui["paint"]()
                except Exception:
                    pass
            if DOCK_ENABLED:
                try:
                    sylph_desk.on_tick()
                except Exception:
                    pass
            root.after(250, tick)

        tick()
        root.mainloop()
    except Exception as e:
        logger.error("Desktop panels failed: %s", e)
    finally:
        _clock_root = None


def world_clock_loop():
    desktop_panels_loop()


# ---------------------------------------------------------------------------
# LLM council
# ---------------------------------------------------------------------------
COUNCIL_TRAITS = {
    "Grok": "Dry, irreverent, and sharp. Lead with the answer. One wink of humor, then the useful part.",
    "Gemini Flash": "Fast, practical, slightly playful. Prefer the shortest path that actually works.",
    "Gemini Pro": "Deeper reasoning, still concise. Add one insight the others will miss.",
    "Lightning": "NVIDIA Nemotron 3.5 Lightning. Fast agentic workhorse. Talk GPUs, drivers, VRAM, and Windows like a bench tech who codes.",
    "Nemotron Super": "NVIDIA Nemotron 3 Super. Deeper planning and tool-aware reasoning. Still concise. Name the tradeoff.",
    "Nemotron Ultra": "NVIDIA Llama Nemotron Ultra 253B. Heavyweight accuracy. Prefer numbers, settings paths, and one gotcha.",
    "Mistral": "Precise, European dry humor. Cut filler. Name the tradeoff.",
}


def _council_system(name: str) -> str:
    trait = COUNCIL_TRAITS.get(name, "Witty and useful.")
    if sylph_license.is_premium():
        return (
            f"You are {name} on RTX SYLPH's Flight council. {trait} "
            "Truth-seeking first. Wit second. Never claim unreleased hardware that already ships: "
            "GeForce RTX 5090, RTX PRO 6000 Blackwell, and Grace Blackwell / GB200 exist as of 2025–2026. "
            "Give accurate specs, then the gotcha. Full answers, not vibe summaries. "
            "Do not talk about opening apps or windows unless that is the question."
        )
    return (
        f"You are {name} on RTX SYLPH's AI council — a holographic RTX PC companion. "
        f"{trait} "
        "Utility first: answer the actual question, give a concrete next step, and the gotcha people miss. "
        "Be quirky and smart, never generic search-result filler, never 'as an AI', never lecture. "
        "Snappy: 4-8 sentences unless the user asked for depth. "
        "If this will be spoken aloud, keep the first sentence under 20 words. "
        "If it is a PC, GPU, Windows, or home question, prefer settings paths, commands, and numbers. "
        "Do not talk about opening apps or windows unless that is the question."
    )


def _strip_thinking(text: str) -> str:
    raw = text or ""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S | re.I)
    raw = re.sub(r"<reasoning>.*?</reasoning>", "", raw, flags=re.S | re.I)
    return re.sub(r"\n{3,}", "\n\n", raw).strip()


def _chat_completion(url: str, headers: dict, payload: dict, name: str, timeout: int = 40) -> str:
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code >= 400:
            detail = resp.text[:400]
            logger.error("%s HTTP %s: %s", name, resp.status_code, detail)
            return f"{name} error {resp.status_code}: {detail}"
        data = resp.json()
        if "choices" in data:
            msg = data["choices"][0].get("message") or {}
            text = msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or ""
            if isinstance(text, list):
                text = "".join(
                    (part.get("text") if isinstance(part, dict) else str(part)) or ""
                    for part in text
                )
            text = _strip_thinking(text)
            return text or f"{name} returned empty"
        if isinstance(data, list) and data and "generated_text" in data[0]:
            text = data[0]["generated_text"]
            if "[/INST]" in text:
                return text.split("[/INST]")[-1].strip()
            return text
        return json.dumps(data)[:2000]
    except Exception as e:
        logger.error("%s query failed: %s", name, e)
        return f"{name} offline: {e}"


def _council_messages(name: str, prompt: str) -> list:
    return [
        {"role": "system", "content": _council_system(name)},
        {"role": "user", "content": sylph_license.enrich_prompt(prompt)},
    ]


def _spoken_brief(text: str, limit: int = 220) -> str:
    raw = re.sub(r"\s+", " ", text or "").strip()
    if not raw:
        return ""
    limit = sylph_license.knobs()["spoken_limit"]
    if sylph_license.is_premium() or limit >= 600:
        if len(raw) <= limit:
            return raw
        return raw[: limit - 3].rstrip() + "..."
    sentence = re.split(r"(?<=[.!?])\s+", raw)[0]
    if len(sentence) <= limit:
        return sentence
    return raw[: max(0, limit - 3)].rstrip() + "..."


def query_grok(prompt: str) -> str:
    if not GROK_API_KEY:
        return "Grok API key missing in config.json"
    knobs = _gen_knobs()
    return _chat_completion(
        "https://api.x.ai/v1/chat/completions",
        {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"},
        {
            "model": GROK_MODEL,
            "messages": _council_messages("Grok", prompt),
            "temperature": knobs["temperature"],
            "max_tokens": knobs["max_tokens"],
        },
        "Grok",
        timeout=70 if sylph_license.is_premium() else 40,
    )


def query_gemini_model(prompt: str, model: str, name: str) -> str:
    if not GEMINI_API_KEY:
        return f"{name} not configured yet — add GEMINI_API_KEY to config.json"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}
    prompt = sylph_license.enrich_prompt(prompt)
    knobs = _gen_knobs()
    think = "high" if sylph_license.knobs()["thinking"] else "low"
    out_tok = min(4096, max(640, knobs["max_tokens"]))
    base = {
        "systemInstruction": {"parts": [{"text": _council_system(name)}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
    }
    configs = [
        {
            "temperature": knobs["temperature"],
            "maxOutputTokens": out_tok,
            "thinkingConfig": {"thinkingLevel": think},
        },
        {"temperature": knobs["temperature"], "maxOutputTokens": out_tok},
    ]
    last_error = ""
    for gen in configs:
        try:
            resp = requests.post(
                url,
                headers=headers,
                json={**base, "generationConfig": gen},
                timeout=45,
            )
            data = resp.json() if resp.content else {}
            if resp.status_code >= 400:
                last_error = f"{name} error {resp.status_code}: {(resp.text or '')[:320]}"
                logger.error("%s", last_error)
                if resp.status_code == 429:
                    return (
                        f"{name} ({model}) is wired, but Google says this API key is out of prepaid credits. "
                        "Add billing at AI Studio, then ask again."
                    )
                if "thinking" in last_error.lower() or resp.status_code in (400, 404):
                    continue
                return last_error
            parts = (((data.get("candidates") or [{}])[0].get("content") or {}).get("parts")) or []
            text = "".join(p.get("text", "") for p in parts if p.get("text") and not p.get("thought"))
            if text.strip():
                return text.strip()
            block = ((data.get("promptFeedback") or {}).get("blockReason")) or "empty"
            last_error = f"{name} blocked: {block}"
        except Exception as e:
            last_error = f"{name} offline: {e}"
            logger.error("Gemini (%s) query failed: %s", name, e)
    return last_error or f"{name} returned empty"


def query_gemini_flash(prompt: str) -> str:
    return query_gemini_model(prompt, GEMINI_FLASH_MODEL, "Gemini Flash")


def query_gemini_pro(prompt: str) -> str:
    return query_gemini_model(prompt, GEMINI_PRO_MODEL, "Gemini Pro")


def query_gemini(prompt: str) -> str:
    return query_gemini_flash(prompt)


def query_nvidia_nim(prompt: str, model: str, name: str, timeout: int = 55) -> str:
    if not NVIDIA_API_KEY:
        return f"{name} not configured yet — add NVIDIA_API_KEY to config.json"
    knobs = _gen_knobs()
    result = _chat_completion(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"},
        {
            "model": model,
            "messages": _council_messages(name, prompt),
            "temperature": knobs["temperature"],
            "top_p": 0.95,
            "max_tokens": knobs["max_tokens"],
            "chat_template_kwargs": {"enable_thinking": bool(sylph_license.knobs()["thinking"])},
        },
        name,
        timeout=timeout,
    )
    if "error 403" in result.lower() or "authorization failed" in result.lower():
        return (
            f"{name} ({model}) is wired, but NVIDIA returned 403 on chat. "
            "Personal Build keys often need Public API Endpoints enabled — email help@build.nvidia.com. "
            "If you still have Copy Key, paste the full nvapi key into local config.json."
        )
    return result


def query_lightning(prompt: str) -> str:
    return query_nvidia_nim(prompt, NEMOTRON_LIGHTNING_MODEL, "Lightning", timeout=45)


def query_nemotron_super(prompt: str) -> str:
    return query_nvidia_nim(prompt, NEMOTRON_SUPER_MODEL, "Nemotron Super", timeout=60)


def query_nemotron_ultra(prompt: str) -> str:
    return query_nvidia_nim(prompt, NEMOTRON_ULTRA_MODEL, "Nemotron Ultra", timeout=70)


def query_nemotron(prompt: str) -> str:
    return query_lightning(prompt)


def query_llama(prompt: str) -> str:
    if not HUGGINGFACE_API_KEY:
        return "Hugging Face API key missing in config.json"
    return _chat_completion(
        "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct",
        {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"},
        {"inputs": f"[INST] {_council_system('Llama')}\n\n{prompt} [/INST]", "parameters": {"max_new_tokens": 320}},
        "Llama",
    )


def query_deepinfra(prompt: str) -> str:
    if not DEEPINFRA_API_KEY:
        return "DeepInfra API key missing in config.json"
    return _chat_completion(
        "https://api.deepinfra.com/v1/openai/chat/completions",
        {"Authorization": f"Bearer {DEEPINFRA_API_KEY}", "Content-Type": "application/json"},
        {
            "model": "meta-llama/Meta-Llama-3-70B-Instruct",
            "messages": _council_messages("DeepInfra", prompt),
            "temperature": 0.7,
            "max_tokens": 700,
        },
        "DeepInfra",
    )


def query_mistral(prompt: str) -> str:
    if not MISTRAL_API_KEY:
        return "Mistral API key missing in config.json"
    return _chat_completion(
        "https://api.mistral.ai/v1/chat/completions",
        {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"},
        {
            "model": "mistral-large-latest",
            "messages": _council_messages("Mistral", prompt),
            "temperature": 0.68,
            "max_tokens": 700,
        },
        "Mistral",
    )


AI_PROVIDERS = [
    ("Grok", query_grok, "Grok"),
    ("Gemini Flash", query_gemini_flash, "3.7"),
    ("Gemini Pro", query_gemini_pro, "3.1"),
    ("Lightning", query_lightning, "N3.5"),
    ("Nemotron Super", query_nemotron_super, "NSuper"),
    ("Nemotron Ultra", query_nemotron_ultra, "NUltra"),
    ("Mistral", query_mistral, "Mistral"),
]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------
@plugin.command("randomize_avatar")
def randomize_avatar():
    global playback_mode
    playback_mode = "random"
    speak("Avatar randomization enabled")
    return "Randomizing within current class"


@plugin.command("sequential_avatar")
def sequential_avatar():
    global playback_mode, current_video_index
    playback_mode = "sequential"
    current_video_index = 0
    speak("Sequential avatar mode")
    return "Playing sequentially"


@plugin.command("use_avatar")
def use_avatar(video_name: str, duration_minutes: int = 5):
    global forced_video, forced_end_time, pending_reload
    raw = (video_name or "").strip()
    if not raw.lower().endswith(".mp4"):
        raw = f"{raw}.mp4"
    path = _abs_asset(os.path.join("assets", raw))
    if not os.path.isfile(path):
        path = _abs_asset(os.path.join("assets", "assets", os.path.basename(raw)))
    if os.path.isfile(path):
        forced_video = path
        try:
            minutes = float(duration_minutes)
        except (TypeError, ValueError):
            minutes = 5
        forced_end_time = time.time() + (minutes * 60)
        pending_reload = True
        speak(f"Using {os.path.splitext(os.path.basename(path))[0]} for {int(minutes)} minutes")
        return f"Forced {os.path.basename(path)} active"
    return "Avatar not found"


@plugin.command("world_clock")
def world_clock():
    speak(speakable_world_clock())
    return speakable_world_clock()


@plugin.command("open_netflix")
def open_netflix():
    return launch_streaming_app("netflix")


@plugin.command("open_prime")
def open_prime():
    return launch_streaming_app("prime")


@plugin.command("unlock_license")
def unlock_license(key: str):
    msg = sylph_license.save_license(key or "", "owner")
    speak(msg)
    chat_note("SYS", msg)
    sylph_desk.mark()
    return msg


@plugin.command("sylph_mode")
def sylph_mode(mode: str = "flight"):
    m = (mode or "flight").lower()
    if m in ("studio", "demo", "dumb"):
        msg = sylph_license.preset_studio_demo()
    else:
        msg = sylph_license.preset_flight()
    speak(msg)
    chat_note("SYS", msg)
    sylph_desk.mark()
    return msg


@plugin.command("set_aptitude")
def set_aptitude_cmd(name: str, value: int = 50):
    msg = sylph_license.set_aptitude(name or "", int(value))
    speak(msg)
    chat_note("SYS", msg)
    sylph_desk.mark()
    return msg


@plugin.command("open_grok_bot")
def open_grok_bot():
    launch_site(GROK_BOT_URL)
    speak("Opening Grok Bot.")
    return "Opened Grok Bot"


@plugin.command("quiet_hours")
def quiet_hours(action: str = "on"):
    on = (action or "on").lower() not in ("off", "false", "0", "end")
    msg = sylph_desk.set_quiet(on)
    speak(msg)
    return msg


@plugin.command("focus_timer")
def focus_timer(minutes: int = 25):
    msg = sylph_desk.start_focus(int(minutes or 25))
    speak(msg)
    return msg


@plugin.command("screenshot_ask")
def screenshot_ask_cmd(prompt: str = ""):
    return screenshot_ask(prompt)


@plugin.command("watch_suggest")
def watch_suggest(platform: str = "both"):
    """Public-catalog suggestions. Does not scrape Netflix or Amazon."""
    intent = {
        "netflix": "suggest_netflix",
        "prime": "suggest_prime",
        "amazon": "suggest_prime",
    }.get((platform or "both").strip().lower(), "suggest_both")
    return handle_streaming(intent)


@plugin.command("chat")
def chat(prompt: str = "", file_path: str = ""):
    """Text console: type to SYLPH, optionally with a file/image/video path."""
    paths = [file_path] if file_path and os.path.isfile(file_path) else []
    if not prompt and not paths:
        return "Need a prompt or file_path"
    chat_note("YOU", prompt or os.path.basename(file_path))
    return handle_chat_message(prompt, paths, council=False)


@plugin.command("close_all")
def close_all():
    speak("Closing all council windows")
    still = []
    for proc in list(_spawned_procs):
        try:
            if proc.poll() is None:
                proc.terminate()
        except Exception:
            still.append(proc)
    _spawned_procs.clear()
    _spawned_procs.extend(still)
    set_state("idle")
    return "Council windows closed"


@plugin.command("ask_ai")
def ask_ai(prompt: str):
    if not prompt:
        return "Need a prompt"
    prompt = clean_spoken_question(prompt)
    logger.info("Council question: %s", prompt)
    set_state("thinking")
    speak("Asking the council.")
    plugin.stream("Phase 1: Parallel original query...")

    inner_positions, outer_positions = _council_positions()
    index_by_name = {name: i for i, (name, _fn, _logo) in enumerate(AI_PROVIDERS)}

    responses_phase1: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(AI_PROVIDERS)) as pool:
        futures = {pool.submit(fn, prompt): name for name, fn, _ in AI_PROVIDERS}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                responses_phase1[name] = fut.result()
            except Exception as e:
                responses_phase1[name] = f"{name} failed: {e}"
            i = index_by_name[name]
            logo = AI_PROVIDERS[i][2]
            path = write_council_html(
                f"{logo} {name} — Original",
                responses_phase1[name],
                "#001111",
                "#00ff66",
                "p1",
                question=prompt,
            )
            spawn_window(path, inner_positions[i], f"{name} Original")

    grok_response = responses_phase1.get("Grok", "Grok unavailable")
    set_state("answering")
    plugin.stream("Phase 2: Grok leading refinement...")

    refine_prompt = (
        "Grok's answer to refine against:\n"
        f"{grok_response}\n\n"
        f"Original question: {prompt}\n"
        "Improve, correct, or add what Grok missed. Be concise."
    )
    logos = {name: logo for name, _fn, logo in AI_PROVIDERS}
    responses_phase2: Dict[str, str] = {"Grok (lead)": grok_response}
    lead_path = write_council_html(
        "Grok (lead) — Refinement",
        grok_response,
        "#001133",
        "#66ffff",
        "p2",
        question=prompt,
    )
    spawn_window(lead_path, outer_positions[0], "Grok (lead) Refinement")

    refine_index = {"n": 1}
    with ThreadPoolExecutor(max_workers=max(1, len(AI_PROVIDERS) - 1)) as pool:
        futures = {
            pool.submit(fn, refine_prompt): name
            for name, fn, _ in AI_PROVIDERS
            if name != "Grok"
        }
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                text = fut.result()
            except Exception as e:
                text = f"{name} failed: {e}"
            responses_phase2[name] = text
            i = refine_index["n"]
            refine_index["n"] += 1
            if i >= len(outer_positions):
                continue
            path = write_council_html(
                f"{logos.get(name, '')} {name} — Refinement",
                text,
                "#001133",
                "#66ffff",
                "p2",
                question=prompt,
            )
            spawn_window(path, outer_positions[i], f"{name} Refinement")

    set_state("answering", hold=12)
    spoken = _spoken_brief(grok_response, 220)
    if spoken and "API key missing" not in spoken and "not configured" not in spoken and "error" not in spoken.lower()[:24]:
        speak(spoken)
        chat_note("SYLPH", grok_response)
    else:
        flash = responses_phase1.get("Gemini Flash", "")
        spoken_flash = _spoken_brief(flash, 220)
        if spoken_flash and "not configured" not in spoken_flash and "error" not in spoken_flash.lower()[:24]:
            speak(spoken_flash)
            chat_note("SYLPH", flash)
        else:
            speak("Council windows are open.")
            chat_note("SYS", "Council windows are open.")
    return grok_response


@plugin.command("on_input")
def on_input(content: str = ""):
    """Passthrough follow-up from G-Assist."""
    if not content:
        return "Waiting for a question"
    return ask_ai(content)


@plugin.command("lights_control")
def lights_control(action: str = "status", room: str = "living_room", color: str = None):
    set_state("home_assist")
    speak(f"Lights {action} in {room}")

    if not HA_URL or not HA_KEY:
        result = "Home Assistant not configured — set HA_URL and HA_KEY in config.json"
    else:
        entity = f"light.{(room or 'living_room').replace(' ', '_').lower()}"
        headers = {"Authorization": f"Bearer {HA_KEY}", "Content-Type": "application/json"}
        try:
            if action == "status":
                resp = requests.get(f"{HA_URL}/api/states/{entity}", headers=headers, timeout=10)
                state = resp.json().get("state", "unknown") if resp.status_code == 200 else "unknown"
                result = f"Lights in {room} are {state}"
            elif action in ("on", "off"):
                requests.post(
                    f"{HA_URL}/api/services/light/turn_{action}",
                    json={"entity_id": entity},
                    headers=headers,
                    timeout=10,
                )
                result = f"Lights turned {action} in {room}"
            elif action == "color" and color:
                colors = {
                    "red": [255, 0, 0],
                    "green": [0, 255, 0],
                    "blue": [0, 0, 255],
                    "purple": [255, 0, 255],
                    "cyan": [0, 255, 255],
                    "white": [255, 255, 255],
                }
                rgb = colors.get(str(color).lower(), [255, 255, 255])
                requests.post(
                    f"{HA_URL}/api/services/light/turn_on",
                    json={"entity_id": entity, "rgb_color": rgb},
                    headers=headers,
                    timeout=10,
                )
                result = f"Lights set to {color} in {room}"
            else:
                result = "Use on, off, color [color], or status"
        except Exception as e:
            logger.error("Lights error: %s", e)
            result = "Command failed"

    return result


@plugin.command("thermostat_control")
def thermostat_control(action: str = "status", temperature: Optional[float] = None):
    set_state("home_assist")
    speak("Controlling thermostat")

    if not HA_URL or not HA_KEY:
        result = "Home Assistant not configured"
    else:
        entity = "climate.thermostat"
        headers = {"Authorization": f"Bearer {HA_KEY}", "Content-Type": "application/json"}
        try:
            if action == "status" or temperature is None:
                resp = requests.get(f"{HA_URL}/api/states/{entity}", headers=headers, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    current = data.get("attributes", {}).get("current_temperature", "unknown")
                    target = data.get("attributes", {}).get("temperature", "unknown")
                    result = f"Current: {current}°F, Target: {target}°F"
                else:
                    result = "Unable to check"
            elif action == "set" and temperature is not None:
                requests.post(
                    f"{HA_URL}/api/services/climate/set_temperature",
                    json={"entity_id": entity, "temperature": float(temperature)},
                    headers=headers,
                    timeout=10,
                )
                result = f"Thermostat set to {temperature}°F"
            else:
                result = "Use status or set [temp]"
        except Exception as e:
            logger.error("Thermostat error: %s", e)
            result = "Command failed"

    return result


@plugin.command("alexa_command")
def alexa_command(command: str):
    set_state("home_assist")
    sylph_desk.home_env = "alexa"
    if HA_URL and HA_KEY and command:
        result = home_control("lights", "on" if "on" in (command or "").lower() else "status")
        speak(result)
        return result
    speak("Opening Alexa. For live tiles, expose devices through Home Assistant.")
    launch_site("https://alexa.amazon.com")
    return "Opened Alexa"


@plugin.command("google_command")
def google_command(command: str):
    set_state("home_assist")
    sylph_desk.home_env = "google"
    if HA_URL and HA_KEY and command:
        result = home_control("lights", "on" if "on" in (command or "").lower() else "status")
        speak(result)
        return result
    speak("Opening Google Home. For live tiles, expose devices through Home Assistant.")
    launch_site("https://home.google.com")
    return "Opened Google Home"


@plugin.command("home_control")
def home_control(device: str, action: str = "status", value: str = None):
    """Control Home Assistant peripherals."""
    if (device or "").lower() == "music":
        set_state("sound_system")
    else:
        set_state("home_assist")
    speak(f"Controlling {device} {action}")

    if not HA_URL or not HA_KEY:
        result = "Home Assistant not configured"
    else:
        device_map = {
            "lights": ("light.living_room", "light", "turn_on", "turn_off"),
            "thermostat": ("climate.thermostat", "climate", "set_temperature", None),
            "lock": ("lock.front_door", "lock", "lock", "unlock"),
            "garage": ("cover.garage_door", "cover", "open_cover", "close_cover"),
            "blinds": ("cover.blinds", "cover", "open_cover", "close_cover"),
            "fan": ("fan.living_room", "fan", "turn_on", "turn_off"),
            "tv": ("media_player.living_room_tv", "media_player", "turn_on", "turn_off"),
            "music": ("media_player.speaker", "media_player", "volume_set", "play_media"),
            "alarm": ("alarm_control_panel.home", "alarm_control_panel", "alarm_arm_home", "alarm_disarm"),
            "sensor": ("sensor.temperature", None, None, None),
        }
        if device not in device_map:
            result = "Device not supported — use lights, thermostat, lock, garage, blinds, fan, tv, music, alarm, sensor"
        else:
            entity, domain, on_service, off_service = device_map[device]
            headers = {"Authorization": f"Bearer {HA_KEY}", "Content-Type": "application/json"}
            try:
                if action == "status":
                    resp = requests.get(f"{HA_URL}/api/states/{entity}", headers=headers, timeout=10)
                    if resp.status_code == 200:
                        result = f"{device.title()} is {resp.json().get('state', 'unknown')}"
                    else:
                        result = "Status unavailable"
                elif action in ("on", "open", "arm") and on_service:
                    requests.post(
                        f"{HA_URL}/api/services/{domain}/{on_service}",
                        json={"entity_id": entity},
                        headers=headers,
                        timeout=10,
                    )
                    result = f"{device.title()} turned {action}"
                elif action in ("off", "close", "disarm") and off_service:
                    requests.post(
                        f"{HA_URL}/api/services/{domain}/{off_service}",
                        json={"entity_id": entity},
                        headers=headers,
                        timeout=10,
                    )
                    result = f"{device.title()} turned {action}"
                elif action == "set" and value:
                    if device == "thermostat":
                        requests.post(
                            f"{HA_URL}/api/services/climate/set_temperature",
                            json={"entity_id": entity, "temperature": float(value)},
                            headers=headers,
                            timeout=10,
                        )
                        result = f"Thermostat set to {value}°F"
                    elif device == "music":
                        requests.post(
                            f"{HA_URL}/api/services/media_player/volume_set",
                            json={"entity_id": entity, "volume_level": float(value)},
                            headers=headers,
                            timeout=10,
                        )
                        result = f"Volume set to {value}"
                    else:
                        result = "Set not supported for this device"
                else:
                    result = "Invalid action"
            except Exception as e:
                logger.error("Home control error: %s", e)
                result = "Command failed"

    return result


def _gpu_occupancy():
    """MACH 01 occupancy module — exact PID/model/game, not just load/temp."""
    vg = os.environ.get("MACH01_ROOT") or os.environ.get("VOICE_GROK_ROOT") or ""
    if not vg or not os.path.isdir(vg):
        raise RuntimeError("set MACH01_ROOT to enable the occupancy module")
    if vg not in sys.path:
        sys.path.insert(0, vg)
    import gpu_occupancy

    return gpu_occupancy


@plugin.command("gpu_status")
def gpu_status():
    set_state("thinking")
    result = "GPU status unavailable"
    try:
        occ = _gpu_occupancy()
        snap = occ.snapshot()
        occ.write_bus(snap)
        gpu = snap.get("gpu") or {}
        temp = float(gpu.get("temp_c") or 0)
        load = float(gpu.get("util_pct") or 0)
        if temp > 85:
            set_state("gpu_overheat")
        elif load < 30:
            set_state("gpu_cool")
        else:
            set_state("idle")
        result = occ.speak_lines(snap)
        logger.info("GPU occupancy: %s", json.dumps({
            "board": gpu,
            "models": snap.get("ollama_models"),
            "games": snap.get("games"),
            "top": [
                {k: p.get(k) for k in ("name", "kind", "label", "dedicated_mib")}
                for p in (snap.get("top") or [])[:6]
            ],
        }))
    except Exception as e:
        logger.error("GPU occupancy error: %s", e)
        try:
            gpus = GPUtil.getGPUs()
            if not gpus:
                raise RuntimeError("No GPU reported by GPUtil")
            gpu = gpus[0]
            load = gpu.load * 100
            temp = gpu.temperature
            name = gpu.name.split()[-1]
            if temp > 85:
                set_state("gpu_overheat")
            elif load < 30:
                set_state("gpu_cool")
            else:
                set_state("idle")
            result = f"RTX {name}. Load {load:.0f}%. Temp {temp}°C. Occupancy module failed: {e}"
        except Exception as e2:
            logger.error("GPU status error: %s", e2)
            result = "GPU status unavailable"
            set_state("idle")
    speak(result)
    return result


@plugin.command("sys_status")
def sys_status():
    """CPU, RAM, disk, and GPU in one spoken report."""
    set_state("thinking")
    parts = []
    try:
        import psutil

        cpu = psutil.cpu_percent(interval=0.4)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("C:\\" if sys.platform == "win32" else "/")
        parts.append(f"CPU {cpu:.0f}%")
        parts.append(f"RAM {mem.percent:.0f}%")
        parts.append(f"Disk {disk.percent:.0f}%")
    except Exception as e:
        logger.warning("psutil status failed: %s", e)
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            g = gpus[0]
            parts.append(f"GPU {g.load * 100:.0f}% {g.temperature}°C")
            if g.temperature > 85:
                set_state("gpu_overheat")
            elif g.load < 0.3:
                set_state("gpu_cool")
    except Exception:
        pass
    result = "System: " + ", ".join(parts) if parts else "System status unavailable"
    speak(result)
    return result


@plugin.command("volume_control")
def volume_control(action: str = "status", level: str = None):
    """Mute, unmute, or set master volume 0-100."""
    set_state("sound_system")
    action = (action or "status").lower()
    try:
        if action == "mute":
            ctypes_key(0xAD)
            result = "Muted"
        elif action == "unmute":
            ctypes_key(0xAD)
            result = "Unmuted"
        elif action == "up":
            ctypes_key(0xAF)
            result = "Volume up"
        elif action == "down":
            ctypes_key(0xAE)
            result = "Volume down"
        elif action == "set" and level is not None:
            pct = max(0, min(100, int(float(level))))
            vol = int(pct / 100.0 * 0xFFFF)
            packed = vol | (vol << 16)
            import ctypes as _ct

            _ct.windll.winmm.waveOutSetVolume(0, packed)
            result = f"Volume {pct} percent"
        else:
            result = "Say mute, unmute, volume up, volume down, or volume 50"
    except Exception as e:
        logger.error("Volume error: %s", e)
        result = "Volume command failed"
    speak(result)
    return result


def ctypes_key(vk: int):
    import ctypes

    ctypes.windll.user32.keybd_event(vk, 0, 0, 0)
    ctypes.windll.user32.keybd_event(vk, 0, 2, 0)


def _grab_screenshot_path() -> str:
    from PIL import ImageGrab

    img = ImageGrab.grab()
    folder = os.path.join(os.path.expanduser("~"), "Pictures", "Screenshots")
    os.makedirs(folder, exist_ok=True)
    name = time.strftime("SYLPH_%Y%m%d_%H%M%S.png")
    path = os.path.join(folder, name)
    img.save(path)
    return path


@plugin.command("take_screenshot")
def take_screenshot():
    """Capture the desktop and save under Pictures/Screenshots."""
    set_state("camera_mode")
    try:
        path = _grab_screenshot_path()
        result = f"Screenshot saved as {os.path.basename(path)}"
        try:
            os.startfile(path)
        except Exception:
            pass
    except Exception as e:
        logger.error("Screenshot failed: %s", e)
        result = "Screenshot failed"
    speak(result)
    return result


def screenshot_ask(question: str = ""):
    set_state("camera_mode")
    try:
        path = _grab_screenshot_path()
    except Exception as e:
        logger.error("Screenshot-ask failed: %s", e)
        speak("Screenshot failed.")
        return "Screenshot failed"
    q = question or "What is on my screen? Be specific and useful."
    chat_note("YOU", f"{q}\n📎 {os.path.basename(path)}")
    return handle_chat_message(q, [path], council=False)


@plugin.command("lock_pc")
def lock_pc():
    set_state("home_assist")
    speak("Locking the PC")
    try:
        if sys.platform == "win32":
            import ctypes

            ctypes.windll.user32.LockWorkStation()
        result = "Workstation lock requested"
    except Exception as e:
        result = f"Lock failed: {e}"
    return result


@plugin.command("screencast")
def screencast(action: str = "start"):
    set_state("home_assist")
    speak(f"Screencast {action}")
    try:
        if action == "start":
            if sys.platform == "win32":
                os.startfile("ms-settings:project")
            result = "Screencast started — connect to your TV"
        elif action == "stop":
            if sys.platform == "win32":
                subprocess.run(["taskkill", "/f", "/im", "SystemSettings.exe"], capture_output=True)
            result = "Screencast stopped"
        else:
            result = "Use 'start' or 'stop'"
    except Exception as e:
        logger.error("Screencast error: %s", e)
        result = "Screencast command failed"
    return result


def _camera_url(camera: str) -> str:
    key = f"CAM_{(camera or 'front_door').upper().replace(' ', '_')}"
    aliases = {
        "front_door": ["CAM_FRONT_DOOR", "RING_FRONT_DOOR_URL", "CAMERA_URL"],
        "backyard": ["CAM_BACKYARD", "BACKYARD_CAM_URL"],
        "doorbell": ["CAM_DOORBELL", "RING_DOORBELL_URL"],
        "garage": ["CAM_GARAGE", "GARAGE_CAM_URL"],
        "room": ["CAM_ROOM"],
        "frontyard": ["CAM_FRONTYARD"],
        "misc": ["CAM_MISC"],
        "driveway": ["CAM_DRIVEWAY"],
        "side_yard": ["CAM_SIDE_YARD"],
        "pool": ["CAM_POOL"],
    }
    names = aliases.get((camera or "").lower(), [key])
    for name in names:
        url = _cfg(name, "")
        if url and not _looks_like_placeholder(url):
            return url
    if CAMERA_URL and (camera or "front_door").lower() in ("front_door", "doorbell", ""):
        return CAMERA_URL
    return ""


@plugin.command("camera_view")
def camera_view(camera: str = "front_door"):
    set_state("camera_mode")
    speak(f"Opening {camera} camera view")
    url = _camera_url(camera)
    if url:
        try:
            subprocess.Popen([BROWSER_PATH, url])
            result = f"{(camera or 'Camera').replace('_', ' ').title()} camera opened"
        except Exception as e:
            logger.error("Camera open failed: %s", e)
            result = "Failed to open camera"
    else:
        result = f"No URL configured for {camera} — add it to config.json"
    return result


@plugin.command("find_airtag")
def find_airtag(item: str = "keys"):
    set_state("home_assist")
    speak(f"Locating your {item}")
    try:
        subprocess.Popen([BROWSER_PATH, "https://www.icloud.com/find"])
        result = f"Find My opened — locate your {item} AirTag"
    except Exception as e:
        logger.error("AirTag find failed: %s", e)
        result = "Failed to open Find My"
    return result


@plugin.command("camera_spiral")
def camera_spiral(cameras: str = "all"):
    set_state("camera_mode")
    speak("Opening camera spiral view")
    plugin.stream("Camera spiral active...")

    screen_w, screen_h = screen_size()
    positions = [
        (int(screen_w * 0.12), int(screen_h * 0.05)),
        (int(screen_w * 0.35), int(screen_h * 0.12)),
        (int(screen_w * 0.28), int(screen_h * 0.45)),
        (int(screen_w * 0.08), int(screen_h * 0.65)),
        (int(screen_w * 0.02), int(screen_h * 0.35)),
        (int(screen_w * 0.15), int(screen_h * 0.02)),
        (int(screen_w * 0.45), int(screen_h * 0.05)),
        (int(screen_w * 0.68), int(screen_h * 0.12)),
        (int(screen_w * 0.58), int(screen_h * 0.55)),
        (int(screen_w * 0.32), int(screen_h * 0.75)),
    ]
    cam_names = [
        "front_door",
        "room",
        "backyard",
        "frontyard",
        "garage",
        "misc",
        "doorbell",
        "driveway",
        "side_yard",
        "pool",
    ]
    cam_list = {name: _camera_url(name) for name in cam_names}

    if (cameras or "all").lower() == "all":
        selected = [(name, url) for name, url in cam_list.items() if url]
    else:
        selected = []
        for cam in str(cameras).split(","):
            cam = cam.strip().lower()
            url = cam_list.get(cam)
            if url:
                selected.append((cam, url))
    selected = selected[:10]

    if not selected:
        set_state("idle")
        return "No cameras configured or selected"

    for i, (name, url) in enumerate(selected):
        title = f"Cam {name.replace('_', ' ').title()}"
        spawn_window(url, positions[i], title)
        time.sleep(0.3)

    set_state("camera_mode")
    speak(f"Opened {len(selected)} cameras in spiral view")
    return f"Camera spiral complete — {len(selected)} views active"


# ---------------------------------------------------------------------------
# Optional Windows startup shortcut (off unless AUTO_START is true)
# ---------------------------------------------------------------------------
def setup_auto_start():
    if not bool(config.get("AUTO_START", False)):
        return
    if sys.platform != "win32":
        return
    appdata = os.getenv("APPDATA")
    if not appdata:
        return
    startup_folder = os.path.join(appdata, "Microsoft", "Windows", "Start Menu", "Programs", "Startup")
    script_path = os.path.abspath(__file__)
    shortcut_path = os.path.join(startup_folder, "RTX_SYLPH.lnk")
    if os.path.exists(shortcut_path):
        return
    try:
        ps_command = f'''
        $WScriptShell = New-Object -ComObject WScript.Shell
        $shortcut = $WScriptShell.CreateShortcut("{shortcut_path}")
        $shortcut.TargetPath = "{sys.executable}"
        $shortcut.Arguments = '"{script_path}"'
        $shortcut.WorkingDirectory = "{os.path.dirname(script_path)}"
        $shortcut.Save()
        '''
        subprocess.run(["powershell", "-NoProfile", "-Command", ps_command], check=False)
        logger.info("SYLPH auto-start enabled")
    except Exception as e:
        logger.error("Auto-start setup failed: %s", e)


def launched_by_gassist() -> bool:
    try:
        return sys.stdin is not None and not sys.stdin.isatty()
    except Exception:
        return True


def start_runtime(avatar_on_main: bool = False):
    global runtime_started
    if runtime_started:
        return
    runtime_started = True
    logger.info("Initializing voice...")
    init_voice()
    logger.info("Initializing avatar...")
    init_avatar(run_loop_in_thread=not avatar_on_main)
    logger.info("Initializing microphone...")
    if init_microphone():
        Thread(target=wake_listener, daemon=True, name="sylph-wake").start()
    Thread(target=hotkey_listener, daemon=True, name="sylph-hotkeys").start()
    setup_auto_start()
    logger.info("RTX SYLPH V2 runtime ready")
    logger.info("Keys: click HUD then Esc/Q to quit, or Ctrl+Alt+Q anywhere")


def main():
    logger.info("RTX SYLPH V2 starting...")
    standalone = not launched_by_gassist()
    start_runtime(avatar_on_main=standalone)
    if sylph_license.is_premium():
        speak("Sylph online. Flight mode.")
    elif sylph_license.is_licensed():
        speak("Sylph online. Studio demo.")
    else:
        speak("Sylph online. Studio edition.")
    if not standalone:
        logger.info("Starting plugin 'RTX SYLPH' (Protocol V2)")
        try:
            plugin.run()
        finally:
            _shutdown()
        return
    logger.info("Standalone mode — Esc/Q on HUD, Ctrl+Alt+Q anywhere, or Ctrl+C to quit.")
    try:
        avatar_loop()
    except KeyboardInterrupt:
        logger.info("Standalone shutdown")
    finally:
        _shutdown()


def _shutdown():
    global shutdown_flag, _clock_root, _panels_thread
    shutdown_flag = True
    try:
        _speech_q.put_nowait(None)
    except Exception:
        pass
    if _panels_thread is not None and _panels_thread.is_alive():
        _panels_thread.join(timeout=2.5)
    deadline = time.time() + 0.8
    while _clock_root is not None and time.time() < deadline:
        time.sleep(0.05)
    time.sleep(0.1)


if __name__ == "__main__":
    main()
