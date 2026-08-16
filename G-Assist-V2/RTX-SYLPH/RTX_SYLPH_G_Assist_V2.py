"""
RTX SYLPH — G-Assist V2 Plugin
Supreme Secure Home Domination Edition
Original concept: @BanditsOfBedlam [Discord] / DailyDriver007 [GitHub]
Powered by Ara @ Colossus Data Center
"""

import os
import sys
import logging
import json
import time
import random
import subprocess
import html as html_lib
import re
from pathlib import Path
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

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
VOICE_SPEED = int(config.get("VOICE_SPEED", 155) or 155)
VOICE_NAME = (_cfg("VOICE_NAME", "zira") or "zira").lower()
GROK_MODEL = _cfg("GROK_MODEL", "grok-4") or "grok-4"
CAMERA_URL = _cfg("CAMERA_URL", "")
VIDEO_PATH = _cfg("VIDEO_PATH", "assets/rtx_sylph_animated.mp4")
FALLBACK_IMAGE_PATH = _cfg("FALLBACK_IMAGE_PATH", "assets/SYLPH_Icon.png")


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
    version="7.7",
    description="Supreme Secure Home Domination Edition",
)

# ---------------------------------------------------------------------------
# Voice
# ---------------------------------------------------------------------------
engine = None
speak_lock = Lock()


def init_voice():
    """SAPI init can hang on Windows. Do it in a worker so SYLPH still starts."""
    global engine
    holder = {"engine": None, "error": None}

    def _init():
        try:
            inst = pyttsx3.init()
            inst.setProperty("rate", VOICE_SPEED)
            wanted = VOICE_NAME
            for voice in inst.getProperty("voices") or []:
                blob = f"{getattr(voice, 'name', '')} {getattr(voice, 'id', '')}".lower()
                if wanted in blob:
                    inst.setProperty("voice", voice.id)
                    logger.info("Voice set to %s", voice.name)
                    break
            holder["engine"] = inst
        except Exception as e:
            holder["error"] = e

    worker = Thread(target=_init, daemon=True, name="sylph-voice-init")
    worker.start()
    worker.join(8)
    if worker.is_alive():
        logger.warning("Voice engine init timed out — continuing without speech")
        engine = None
        return
    if holder["error"] is not None:
        logger.error("Voice engine init failed: %s", holder["error"])
        engine = None
        return
    engine = holder["engine"]


def speak(text: str):
    if not text:
        return
    logger.info("SYLPH: %s", text)
    if engine is None:
        return

    def _say():
        with speak_lock:
            try:
                engine.stop()
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                logger.error("Voice output failed: %s", e)

    worker = Thread(target=_say, daemon=True, name="sylph-speak")
    worker.start()
    worker.join(20)
    if worker.is_alive():
        logger.warning("Speech timed out")


# ---------------------------------------------------------------------------
# Avatar / animation
# ---------------------------------------------------------------------------
AVATAR_W, AVATAR_H = 200, 300
video_lock = Lock()
current_state = "idle"
current_video_index = 0
playback_mode = "sequential"
forced_video = None
forced_end_time = 0.0
state_hold_until = 0.0
state_sticky = False
cap_current = None
last_surface = None
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
    "listening":    {"hold": 12, "next": "idle",        "sticky": False, "overlay": (0, 110, 255, 70)},
    "thinking":     {"hold": 0,  "next": "thinking",    "sticky": True,  "overlay": (0, 220, 255, 55)},
    "answering":    {"hold": 10, "next": "idle",        "sticky": False, "overlay": (0, 255, 120, 55)},
    "gpu_cool":     {"hold": 14, "next": "idle",        "sticky": False, "overlay": (0, 255, 200, 40)},
    "gpu_overheat": {"hold": 14, "next": "idle",        "sticky": False, "overlay": (255, 40, 40, 70)},
    "home_assist":  {"hold": 10, "next": "idle",        "sticky": False, "overlay": (255, 180, 0, 50)},
    "sound_system": {"hold": 10, "next": "idle",        "sticky": False, "overlay": (180, 80, 255, 50)},
    "camera_mode":  {"hold": 10, "next": "idle",        "sticky": False, "overlay": (90, 90, 255, 50)},
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


def build_animation_classes() -> Dict[str, List[str]]:
    idle = discover_videos("sylph_idle", "rtx_sylph_animated")
    extra_idle = _abs_asset(VIDEO_PATH)
    if extra_idle and os.path.isfile(extra_idle) and extra_idle not in idle:
        idle.append(extra_idle)
    classes = {
        "idle": idle,
        "thinking": discover_videos("sylph_thinking"),
        "answering": discover_videos("sylph_answers"),
        "gpu_cool": discover_videos("gpu_cool"),
        "gpu_overheat": discover_videos("gpu_overheat"),
        "home_assist": discover_videos("sylph_home_assist"),
        "sound_system": discover_videos("rtx_sylph_sound_system"),
        "camera_mode": discover_videos("sylph_camera", "sylph_pc_rog"),
        "listening": idle[:],
    }
    for key, videos in list(classes.items()):
        if not videos:
            classes[key] = idle[:]
            logger.warning("No videos for class '%s' — falling back to idle", key)
        else:
            logger.info("Animation class '%s': %d clips", key, len(videos))
    return classes


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
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            cap_current = cap
            logger.info("Loaded: %s (%s)", os.path.basename(video_path), current_state)
            return
        cap.release()

    logger.warning("No playable video for state '%s'", current_state)


def _fit_frame(frame, tw: int, th: int):
    """Letterbox a BGR frame into tw x th without stretching SYLPH."""
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((th, tw, 3), dtype=np.uint8)
    scale = min(tw / w, th / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    x, y = (tw - nw) // 2, (th - nh) // 2
    canvas[y : y + nh, x : x + nw] = resized
    return canvas


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
    global last_surface, cap_current
    with video_lock:
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

        try:
            fitted = _fit_frame(frame, AVATAR_W, AVATAR_H)
            rgb = cv2.cvtColor(fitted, cv2.COLOR_BGR2RGB)
            last_surface = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        except Exception as e:
            logger.error("Frame convert failed: %s", e)
            last_surface = fallback_surface or _blank_surface()
        return last_surface


def set_state(new_state: str, hold: Optional[float] = None):
    """Switch SYLPH's being-state and load that class of Drive clips."""
    global current_state, current_video_index, state_hold_until, state_sticky
    if new_state not in ANIMATION_CLASSES:
        new_state = "idle"
    meta = STATE_META.get(new_state, STATE_META["idle"])
    seconds = meta["hold"] if hold is None else float(hold)
    with video_lock:
        changed = current_state != new_state
        current_state = new_state
        state_sticky = bool(meta.get("sticky"))
        state_hold_until = time.time() + seconds if seconds > 0 else 0.0
        if changed:
            current_video_index = 0
            load_next_video()
            logger.info("State -> %s (hold=%.1fs sticky=%s)", new_state, seconds, state_sticky)


def avatar_loop():
    global last_surface
    clock = pygame.time.Clock()
    try:
        while not shutdown_flag:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            if screen is None:
                time.sleep(0.05)
                continue
            frame = get_avatar_frame()
            tint = STATE_META.get(current_state, {}).get("overlay")
            if tint:
                overlay = pygame.Surface((AVATAR_W, AVATAR_H), pygame.SRCALPHA)
                overlay.fill(tint)
                frame = frame.copy()
                frame.blit(overlay, (0, 0))
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
    """Overlay GPU stats on a *copy* of the last frame. Does not advance video."""
    out_path = os.path.join(_plugin_dir, "cooler_sylph.png")
    font_path = r"C:\Windows\Fonts\arialbd.ttf"
    while not shutdown_flag:
        try:
            with video_lock:
                surf = last_surface
            if surf is None:
                time.sleep(2)
                continue
            pygame.image.save(surf, out_path)
            img = Image.open(out_path).convert("RGBA")
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


def init_avatar():
    global screen, ANIMATION_CLASSES
    ANIMATION_CLASSES = build_animation_classes()
    os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "0,0")
    pygame.init()
    pygame.display.set_caption("RTX SYLPH v7.7")
    screen = pygame.display.set_mode((AVATAR_W, AVATAR_H), pygame.NOFRAME)
    pygame.mouse.set_visible(False)
    load_fallback_surface()
    with video_lock:
        load_next_video()
    Thread(target=avatar_loop, daemon=True, name="sylph-avatar").start()
    Thread(target=cooler_mirror_loop, daemon=True, name="sylph-cooler").start()


# ---------------------------------------------------------------------------
# Wake word
# ---------------------------------------------------------------------------
recognizer = sr.Recognizer()
microphone = None


def init_microphone():
    global microphone
    try:
        microphone = sr.Microphone()
        logger.info("Microphone opened — wake word '%s'", WAKE_WORD)
        return True
    except Exception as e:
        logger.warning("Microphone unavailable (wake word disabled): %s", e)
        microphone = None
        return False


def wake_listener():
    if microphone is None:
        return
    try:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.4)
        logger.info("Wake listener ready — say '%s'", WAKE_WORD)
    except Exception as e:
        logger.warning("Ambient noise calibration failed: %s", e)
    while not shutdown_flag:
        try:
            with microphone as source:
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=3)
            try:
                text = recognizer.recognize_google(audio).lower()
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                logger.warning("Speech recognition request failed: %s", e)
                time.sleep(2)
                continue
            logger.info("Heard: %s", text)
            if WAKE_WORD in text:
                set_state("listening")
                speak("Yes, master?")
        except sr.WaitTimeoutError:
            continue
        except Exception as e:
            logger.error("Wake error: %s", e)
            time.sleep(1)


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------
COUNCIL_DIR = os.path.join(_plugin_dir, "council_windows")


def write_council_html(title: str, body: str, bg: str, fg: str, phase: str) -> str:
    os.makedirs(COUNCIL_DIR, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", title).strip("_")[:80] or "council"
    path = os.path.join(COUNCIL_DIR, f"{phase}_{slug}.html")
    html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<title>{html_lib.escape(title)}</title>"
        "<style>"
        "html,body{margin:0;padding:0;height:100%;}"
        f"body{{font-family:Consolas,'Cascadia Mono',monospace;background:{bg};color:{fg};padding:18px;}}"
        "h1{font-size:15px;letter-spacing:1px;margin:0 0 12px;text-transform:uppercase;}"
        "pre{white-space:pre-wrap;word-wrap:break-word;font-size:13px;line-height:1.45;}"
        "</style></head><body>"
        f"<h1>{html_lib.escape(title)}</h1>"
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
    args = [
        BROWSER_PATH,
        "--new-window",
        "--window-size=520,640",
        f"--window-position={x},{y}",
        f"--app={url}",
    ]
    try:
        proc = subprocess.Popen(args)
        _spawned_procs.append(proc)
        logger.info("Window spawned: %s at (%s,%s) pid=%s", title, x, y, proc.pid)
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


# ---------------------------------------------------------------------------
# LLM council
# ---------------------------------------------------------------------------
def _chat_completion(url: str, headers: dict, payload: dict, name: str, timeout: int = 30) -> str:
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if "choices" in data:
            return data["choices"][0]["message"]["content"]
        if isinstance(data, list) and data and "generated_text" in data[0]:
            text = data[0]["generated_text"]
            if "[/INST]" in text:
                return text.split("[/INST]")[-1].strip()
            return text
        return json.dumps(data)[:2000]
    except Exception as e:
        logger.error("%s query failed: %s", name, e)
        return f"{name} offline: {e}"


def query_grok(prompt: str) -> str:
    if not GROK_API_KEY:
        return "Grok API key missing in config.json"
    return _chat_completion(
        "https://api.x.ai/v1/chat/completions",
        {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"},
        {"model": GROK_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.7},
        "Grok",
    )


def query_chatgpt(prompt: str) -> str:
    if not OPENAI_API_KEY:
        return "ChatGPT API key missing in config.json"
    return _chat_completion(
        "https://api.openai.com/v1/chat/completions",
        {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        {"model": "gpt-4o", "messages": [{"role": "user", "content": prompt}], "temperature": 0.7},
        "ChatGPT",
    )


def query_nemotron(prompt: str) -> str:
    if not NVIDIA_API_KEY:
        return "NVIDIA API key missing in config.json"
    return _chat_completion(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"},
        {
            "model": "nvidia/llama-3.1-nemotron-70b-instruct",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.6,
            "max_tokens": 1024,
        },
        "Nemotron",
    )


def query_llama(prompt: str) -> str:
    if not HUGGINGFACE_API_KEY:
        return "Hugging Face API key missing in config.json"
    return _chat_completion(
        "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct",
        {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"},
        {"inputs": f"[INST] {prompt} [/INST]", "parameters": {"max_new_tokens": 250}},
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
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
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
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        },
        "Mistral",
    )


AI_PROVIDERS = [
    ("Grok", query_grok, "🦔"),
    ("ChatGPT", query_chatgpt, "🤖"),
    ("Nemotron", query_nemotron, "⚡"),
    ("Llama", query_llama, "🦙"),
    ("DeepInfra", query_deepinfra, "🌊"),
    ("Mistral", query_mistral, "🌫️"),
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
    global forced_video, forced_end_time
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
        with video_lock:
            load_next_video()
        speak(f"Using {os.path.splitext(os.path.basename(path))[0]} for {int(minutes)} minutes")
        return f"Forced {os.path.basename(path)} active"
    return "Avatar not found"


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


def _council_positions() -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Spiral to the right of SYLPH's 200x300 HUD in the upper-left."""
    screen_w, screen_h = screen_size()
    # Reserve the HUD corner; first column starts just past 200px.
    inner = [
        (220, 8),
        (760, 8),
        (1300 if screen_w > 1600 else max(220, screen_w - 540), 8),
        (220, 360),
        (760, 360),
        (1300 if screen_w > 1600 else max(220, screen_w - 540), 360),
    ]
    # Phase 2 sits slightly offset so both rings stay visible.
    outer = [(p[0] + 40, p[1] + 30) for p in inner]
    return inner, outer


@plugin.command("ask_ai")
def ask_ai(prompt: str):
    if not prompt:
        return "Need a prompt"
    set_state("thinking")
    speak("Activating AI council")
    plugin.stream("Phase 1: Parallel original query...")

    inner_positions, outer_positions = _council_positions()
    index_by_name = {name: i for i, (name, _fn, _logo) in enumerate(AI_PROVIDERS)}

    responses_phase1: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
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
        "🦔 Grok (lead) — Refinement",
        grok_response,
        "#001133",
        "#66ffff",
        "p2",
    )
    spawn_window(lead_path, outer_positions[0], "Grok (lead) Refinement")

    refine_index = {"n": 1}
    with ThreadPoolExecutor(max_workers=5) as pool:
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
            )
            spawn_window(path, outer_positions[i], f"{name} Refinement")

    set_state("answering", hold=12)
    speak("AI council complete — spiral pattern active")
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
    speak(f"Sending to Alexa: {command}")
    result = "Alexa command sent (integrate with Home Assistant for full control)"
    return result


@plugin.command("google_command")
def google_command(command: str):
    set_state("home_assist")
    speak(f"Sending to Google Home: {command}")
    result = "Google command sent (integrate with Home Assistant for full control)"
    return result


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


@plugin.command("gpu_status")
def gpu_status():
    set_state("thinking")
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
        result = f"RTX {name}. Load {load:.0f}%. Temp {temp}°C."
    except Exception as e:
        logger.error("GPU status error: %s", e)
        result = "GPU status unavailable"
        set_state("idle")
    speak(result)
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


def start_runtime():
    global runtime_started
    if runtime_started:
        return
    runtime_started = True
    logger.info("Initializing voice...")
    init_voice()
    logger.info("Initializing avatar...")
    init_avatar()
    logger.info("Initializing microphone...")
    if init_microphone():
        Thread(target=wake_listener, daemon=True, name="sylph-wake").start()
    setup_auto_start()
    logger.info("RTX SYLPH V2 runtime ready")


def main():
    logger.info("RTX SYLPH V2 starting...")
    start_runtime()
    speak("Sylph online.")
    if launched_by_gassist():
        logger.info("Starting plugin 'RTX SYLPH' (Protocol V2)")
        try:
            plugin.run()
        finally:
            _shutdown()
    else:
        logger.info("Standalone mode — avatar and wake word active. Ctrl+C to quit.")
        try:
            while not shutdown_flag:
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("Standalone shutdown")
        finally:
            _shutdown()


def _shutdown():
    global shutdown_flag
    shutdown_flag = True
    time.sleep(0.2)


if __name__ == "__main__":
    main()
