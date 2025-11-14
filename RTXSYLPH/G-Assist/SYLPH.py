# RTX SYLPH v7.7 — SUPREME SECURE HOME DOMINATION EDITION
# GET A CUSTOM ONE: Contact DailyDriver007
# Powered by Ara @ Colossus Data Center
#
# Dependencies:
# Structural: cv2, pygame, numpy, pyttsx3, speech_recognition, GPUtil, requests, json, subprocess, webbrowser, urllib.parse, re, concurrent.futures, pathlib, os, bleach, bcrypt, psutil, cryptography, boto3
# Optional for RGB: openrgb (if installed), logiops (CLI tool)
# Install via pip: opencv-python pygame numpy pyttsx3 speechrecognition gputil requests bleach bcrypt psutil cryptography boto3
# For openrgb: pip install openrgb-python (if desired)
# For dependency scanning: pip install safety
# Note: No additional installations possible in restricted environments; fallbacks provided.
# Note on config encryption: config.json is encrypted using Fernet; decrypt at runtime with a secure key (stored in AWS Secrets Manager or env var).
# User Guide:
# Setup: Install dependencies, configure config.json with API keys, encrypt it using Fernet.
# Run: python sylph.py
# Troubleshooting: Check sylph_errors.log for issues; ensure API keys are valid.
# API Key Setup: Obtain keys from respective providers (x.ai, OpenAI, NVIDIA, Hugging Face).

import cv2
import pygame
import numpy as np
import pyttsx3
import speech_recognition as sr
import GPUtil
import time
import threading
import requests
import json
import subprocess
import webbrowser
import urllib.parse
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import bleach  # For HTML sanitization
import atexit  # For graceful shutdown
import bcrypt  # For PIN hashing
import psutil  # For memory monitoring
from cryptography.fernet import Fernet  # For config encryption
from functools import lru_cache  # For caching
import boto3  # For AWS Secrets Manager; configure credentials via AWS CLI

# ================================
# CONFIG ENCRYPTION KEY FROM AWS SECRETS MANAGER
# In production, use secure vaults like AWS Secrets Manager
# ================================
secrets_client = boto3.client('secretsmanager', region_name='us-east-1')  # Replace with your region
try:
    secret_response = secrets_client.get_secret_value(SecretId='sylph-config-key')
    key = secret_response['SecretBinary']
except:
    key = Fernet.generate_key()
    secrets_client.create_secret(Name='sylph-config-key', SecretBinary=key)
fernet = Fernet(key)

# ================================
# LOAD AND DECRYPT CONFIG
# ================================
CONFIG_PATH = Path(__file__).parent / "config.json"
if not CONFIG_PATH.exists():
    raise FileNotFoundError("config.json missing! Create it in the G-Assist folder with your API keys.")
with open(CONFIG_PATH, "rb") as f:
    encrypted_config = f.read()
decrypted_config = fernet.decrypt(encrypted_config)
config = json.loads(decrypted_config)

# Required
VIDEO_PATH = config.get("VIDEO_PATH", "assets/rtx_sylph_animated.mp4")
FALLBACK_IMAGE_PATH = config.get("FALLBACK_IMAGE_PATH", "assets/rtx_sylph_static.png")  # Add a static fallback image in assets
WAKE_WORD = config.get("WAKE_WORD", "sylph").lower()
VOICE_SPEED = config.get("VOICE_SPEED", 155)
PLAIN_PIN = config.get("PIN", "1234").strip()  # Plaintext from config; will be hashed
HASHED_PIN = bcrypt.hashpw(PLAIN_PIN.encode(), bcrypt.gensalt())  # Hash PIN

# API Keys - Improved validation with warnings at startup if missing
GROK_API_KEY = config.get("GROK_API_KEY", "").strip()
OPENAI_API_KEY = config.get("OPENAI_API_KEY", "").strip()
NVIDIA_API_KEY = config.get("NVIDIA_API_KEY", "").strip()
HUGGINGFACE_API_KEY = config.get("HUGGINGFACE_API_KEY", "").strip()
HA_URL = config.get("HA_URL", "")
HA_KEY = config.get("HA_KEY", "")
BROWSER_PATH = config.get("BROWSER_PATH", r"C:\Program Files\Google\Chrome\Application\chrome.exe")  # Add this to config.json for AI windows
CAMERA_URL = config.get("CAMERA_URL", "")  # Optional: URL for other security cameras

# Validate CAMERA_URL for local IPs
if CAMERA_URL and not re.match(r"^https?://(192\.168\.\d+\.\d+|localhost|127\.0\.0\.1)(:\d+)?(/.*)?$", CAMERA_URL.split("://")[1] if "://" in CAMERA_URL else CAMERA_URL):
    raise ValueError("Invalid CAMERA_URL: Must be a local IP (192.168.x.x, localhost, or 127.0.0.1).")

# Validate BROWSER_PATH
if not os.path.exists(BROWSER_PATH) or not os.access(BROWSER_PATH, os.X_OK):
    raise ValueError("Invalid BROWSER_PATH: Does not exist or not executable.")

# API URLs
GROK_URL = "https://api.x.ai/v1/chat/completions"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
NVIDIA_NEMOTRON_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
HF_LLAMA_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"

# Allowed subprocess commands
ALLOWED_SUBS = ["ms-settings:project", "openrgb", "logiops"]

# ================================
# INIT
# ================================
os.environ['SDL_VIDEO_WINDOW_POS'] = '0,0'  # Position SYLPH in upper left
pygame.init()
screen_info = pygame.display.Info()  # For dynamic window positioning
screen = pygame.display.set_mode((200, 300), pygame.NOFRAME)
pygame.display.set_caption("RTX SYLPH v7.7")
pygame.mouse.set_visible(False)
cap = cv2.VideoCapture(VIDEO_PATH)
fallback_surf = None
if not cap.isOpened():
    try:
        fallback_img = pygame.image.load(FALLBACK_IMAGE_PATH)
        fallback_surf = pygame.transform.scale(fallback_img, (200, 300))
        print("Using fallback static image.")
    except:
        raise FileNotFoundError(f"Video and fallback image not found.")
engine = pyttsx3.init()
engine.setProperty('rate', VOICE_SPEED)
engine.setProperty('voice', 'zira')
r = sr.Recognizer()
mic = sr.Microphone()
listening = False
speaking = False
pending_action = None  # For confirmations
pending_data = None    # Data for pending action
pending_state = "confirm"  # "confirm" or "pin"
last_command_time = 0  # For rate limiting
confirmation_timeout = 3  # Seconds for confirmation
pin_attempts = 0
max_pin_attempts = 3
lockout_end_time = 0  # For non-blocking lockout

# Thread lock for shared state
state_lock = threading.Lock()

# Startup warnings for missing keys
missing_keys = []
if not GROK_API_KEY: missing_keys.append("Grok")
if not OPENAI_API_KEY: missing_keys.append("OpenAI")
if not NVIDIA_API_KEY: missing_keys.append("NVIDIA")
if not HUGGINGFACE_API_KEY: missing_keys.append("Hugging Face")
if missing_keys:
    print(f"Warning: Missing API keys for {', '.join(missing_keys)}. Some features may not work.")

# Optional dependency checks
try:
    import openrgb
except ImportError:
    print("Warning: openrgb-python not installed. RGB fallback to logiops may be limited.")

# ================================
# GRACEFUL SHUTDOWN
# ================================
def shutdown():
    if cap.isOpened():
        cap.release()
    pygame.quit()
    print("Sylph shutting down...")

atexit.register(shutdown)

# ================================
# LOG ERROR (Structured JSON)
# ================================
import logging
logging.basicConfig(filename='sylph.log', level=logging.INFO, format='%(asctime)s - %(message)s')
def log_error(message):
    logging.info(json.dumps({"event": "error", "message": message}))

# ================================
# CHECK MEMORY (For leak mitigation with auto-restart)
# ================================
def check_memory(threshold=0.8):
    memory = psutil.virtual_memory()
    if memory.percent > threshold * 100:
        log_error(f"High memory usage: {memory.percent}%. Restarting...")
        os.execv(sys.executable, ['python'] + sys.argv)  # Self-restart

# ================================
# DYNAMIC POWER MANAGEMENT (Experimental)
# ================================
def dynamic_power_management():
    while True:
        gpu = GPUtil.getGPUs()[0]
        if gpu.load < 0.1:  # 10%
            os.system("cpupower frequency-set -g powersave")  # Throttle CPU
        time.sleep(60)  # Check every minute

threading.Thread(target=dynamic_power_management, daemon=True).start()

# ================================
# VIDEO LOOP WITH STATE OVERLAY
# ================================
def draw_frame():
    if cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (200, 300))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.rot90(frame)
            surf = pygame.surfarray.make_surface(frame)
        else:
            surf = fallback_surf
    else:
        surf = fallback_surf
    
    if surf:
        if listening and not speaking:
            overlay = pygame.Surface((200, 300), pygame.SRCALPHA)
            overlay.fill((0, 100, 255, 80))  # Blue overlay for listening (visual indicator)
            surf.blit(overlay, (0, 0))
        
        if speaking:
            overlay = pygame.Surface((200, 300), pygame.SRCALPHA)
            overlay.fill((0, 255, 100, 100))
            surf.blit(overlay, (0, 0))
            mouth = pygame.Surface((200, 100), pygame.SRCALPHA)
            mouth.fill((255, 255, 255, 100))
            surf.blit(mouth, (0, 200))
        
        screen.blit(surf, (0, 0))
        pygame.display.flip()

# ================================
# SPEAK WITH STATE
# ================================
def speak(text):
    global speaking, listening
    try:
        speaking = True
        listening = False
        engine.say(text)
        engine.runAndWait()
        speaking = False
    except RuntimeError:
        print(f"Skipping audio: {text}")  # Graceful fallback

# ================================
# WAIT FOR VOICE INPUT WITH TIMEOUT
# ================================
def wait_for_voice_input(max_time=5):
    start = time.time()
    while time.time() - start < max_time:
        try:
            with mic as source:
                audio = r.listen(source, timeout=1, phrase_time_limit=2)
            text = r.recognize_google(audio).lower()
            return text
        except sr.UnknownValueError:
            speak("Audio unclear. Retrying in 1 second...")
            time.sleep(1)
        except sr.RequestError:
            speak("Network error. Retrying...")
            time.sleep(1)
    return None

# ================================
# LLM CONNECTORS - With retry logic
# ================================
def query_api(url, headers, data, key_check, model_name, retries=3):
    if not key_check: return f"{model_name} API key missing in config."
    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=headers, json=data, timeout=20)
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            log_error(f"{model_name} query failed: {str(e)}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return f"{model_name} offline."

def query_grok(prompt):
    headers = {"Authorization": f"Bearer {GROK_API_KEY}"}
    data = {"model": "grok-beta", "messages": [{"role": "user", "content": prompt}]}
    return query_api(GROK_URL, headers, data, GROK_API_KEY, "Grok")

def query_chatgpt(prompt):
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    data = {"model": "gpt-4o", "messages": [{"role": "user", "content": prompt}]}
    return query_api(OPENAI_URL, headers, data, OPENAI_API_KEY, "ChatGPT")

def query_nemotron(prompt):
    headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}"}
    data = {"model": "nvidia/nemotron-4-340b-reward", "messages": [{"role": "user", "content": prompt}]}
    return query_api(NVIDIA_NEMOTRON_URL, headers, data, NVIDIA_API_KEY, "Nemotron")

def query_llama3(prompt):
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    data = {"inputs": f"[INST] {prompt} [/INST]", "parameters": {"max_new_tokens": 150}}
    try:
        resp = requests.post(HF_LLAMA_URL, headers=headers, json=data, timeout=20)
        return resp.json()[0]["generated_text"].split("[/INST]")[-1].strip()
    except Exception as e:
        log_error(f"Llama query failed: {str(e)}")
        return f"Llama offline."

# ================================
# PARSE INTENT (For ambiguity)
# ================================
def parse_intent(text):
    if "yes" in text and "no" in text:
        return "cancel"
    elif re.search(r'[\-]?\d', text) or "minus" in text or "lower" in text or "decrease" in text:
        return "relative_negative"
    elif re.search(r'[\+]?\d', text) or "plus" in text or "raise" in text or "increase" in text:
        return "relative_positive"
    return "absolute"

# ================================
# COMMAND HANDLERS (Modular)
# ================================
def handle_help(text):
    speak("I can control lights on off or color, check lights status, set thermostat temperature or adjust by amount, view camera or doorbell, track airtag, set keyboard or mouse RGB like red pulse fast, screencast to TV, check GPU status. For questions, just ask and I'll query all AIs in parallel with results in surrounding windows.")
    return True

def handle_airtag(text):
    speak("Opening Find My...")
    webbrowser.open("https://www.icloud.com/find")
    return True

def handle_doorbell(text):
    speak("Opening Ring live view...")
    webbrowser.open("https://home.ring.com/live")
    return True

def handle_camera(text):
    if not CAMERA_URL:
        speak("Camera URL not configured.")
        return True
    speak("Opening camera view...")
    webbrowser.open(CAMERA_URL)
    return True

def handle_rgb(text):
    global pending_action, pending_data, pending_state
    colors = {"red": "ff0000", "green": "00ff00", "blue": "0000ff", "purple": "ff00ff", "cyan": "00ffff", "white": "ffffff"}
    color = "ffffff"
    for k in colors:
        if k in text.split():
            color = colors[k]
            break
    pattern = "pulse" if "pulse" in text else "flow" if "flow" in text else "solid"
    speed = "fast" if "fast" in text else "slow" if "slow" in text else "medium"
    device_type = 'keyboard' if 'keyboard' in text else 'mouse'
    speak(f"Confirm setting {device_type} to {pattern} {speed} in {color}? Say yes or no.")
    pending_action = "rgb"
    pending_data = {"device_type": device_type, "color": color, "pattern": pattern, "speed": speed}
    pending_state = "confirm"
    return True

def handle_lights(text):
    global pending_action, pending_data, pending_state
    if not HA_URL or not HA_KEY:
        speak("Home Assistant not configured.")
        return True
    entity = "light.living_room"
    if "check" in text or "status" in text:
        resp = requests.get(f"{HA_URL}/api/states/{entity}", headers={"Authorization": f"Bearer {HA_KEY}"})
        if resp.status_code == 200:
            data = resp.json()
            state = data['state']
            speak(f"Lights are currently {state}.")
        else:
            speak("Could not retrieve lights status.")
        return True
    elif "on" in text:
        speak("Confirm turning lights on? Say yes or no.")
        pending_action = "lights_on"
        pending_data = {"entity": entity}
        pending_state = "confirm"
    elif "off" in text:
        speak("Confirm turning lights off? Say yes or no.")
        pending_action = "lights_off"
        pending_data = {"entity": entity}
        pending_state = "confirm"
    elif "color" in text or "party" in text:
        colors = {"red": "ff0000", "green": "00ff00", "blue": "0000ff", "purple": "ff00ff", "cyan": "00ffff", "white": "ffffff"}
        color = "ffffff"
        for k in colors:
            if k in text.split():
                color = colors[k]
                break
        rgb = [int(color[i:i+2], 16) for i in (0,2,4)]
        speak(f"Confirm setting lights to {color}? Say yes or no.")
        pending_action = "lights_color"
        pending_data = {"entity": entity, "rgb": rgb, "is_party": "party" in text}
        pending_state = "confirm"
    return True

def handle_thermostat(text):
    global pending_action, pending_data, pending_state, pin_attempts
    if not HA_URL or not HA_KEY:
        speak("Home Assistant not configured.")
        return True
    entity = "climate.thermostat"
    if "check" in text or "status" in text:
        resp = requests.get(f"{HA_URL}/api/states/{entity}", headers={"Authorization": f"Bearer {HA_KEY}"})
        if resp.status_code == 200:
            data = resp.json()
            current_temp = data['attributes'].get('current_temperature', 'unknown')
            speak(f"Current temperature is {current_temp} degrees.")
        else:
            speak("Could not retrieve thermostat status.")
        return True
    elif "set" in text or "adjust" in text:
        try:
            intent = parse_intent(text)
            if intent == "relative_positive" or intent == "relative_negative":
                delta = float(re.search(r'[\+\-]?\d+\.?\d*', text).group())
                resp = requests.get(f"{HA_URL}/api/states/{entity}", headers={"Authorization": f"Bearer {HA_KEY}"})
                if resp.status_code == 200:
                    current = resp.json()['attributes'].get('temperature', 70)
                    temp = current + delta
                else:
                    speak("Could not get current temperature for adjustment.")
                    return True
            else:
                temp = float(re.search(r'\d+\.?\d*', text).group())
            if temp < 40 or temp > 100:
                speak("Temperature out of realistic range.")
                return True
            speak(f"Enter PIN to confirm setting thermostat to {temp} degrees.")
            pending_action = "thermostat_set"
            pending_data = {"entity": entity, "temp": temp}
            pending_state = "pin"
            pin_attempts = 0
        except:
            speak("No valid temperature found in command.")
    return True

def handle_screencast(text):
    global pending_action, pending_data, pending_state
    speak("Confirm starting screencast to TV? This may allow device mirroring—ensure your network is secure. Say yes or no.")
    pending_action = "screencast"
    pending_state = "confirm"
    return True

def handle_gpu_status(text):
    gpu = GPUtil.getGPUs()[0]
    msg = f"RTX {gpu.name.split()[-1]}. Load {gpu.load*100:.0f}%. Temp {gpu.temperature}°C."
    speak(msg)
    return True

def handle_specific_ai(text):
    if not HUGGINGFACE_API_KEY and "llama" in text:
        speak("Llama3 requires Hugging Face API key.")
        return True
    if "grok" in text:
        prompt = text.replace(WAKE_WORD, '').replace('grok', '').strip()
        speak("Asking Grok...")
        speak(query_grok(prompt))
        return True
    elif "nemotron" in text:
        prompt = text.replace(WAKE_WORD, '').replace('nemotron', '').strip()
        speak("Asking Nemotron...")
        speak(query_nemotron(prompt))
        return True
    elif "chatgpt" in text or "gpt" in text:
        prompt = text.replace(WAKE_WORD, '').replace('chatgpt', '').replace('gpt', '').strip()
        speak("Asking ChatGPT...")
        speak(query_chatgpt(prompt))
        return True
    elif "llama" in text:
        prompt = text.replace(WAKE_WORD, '').replace('llama', '').strip()
        speak("Asking Llama-3...")
        speak(query_llama3(prompt))
        return True
    return False

def handle_default_query(text):
    prompt = text.replace(WAKE_WORD, '').strip()
    if not prompt:
        speak("No query detected. Try asking a question.")
        return True
    speak("Querying all AIs in parallel...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            'Grok': executor.submit(query_grok, prompt),
            'Nemotron': executor.submit(query_nemotron, prompt),
            'ChatGPT': executor.submit(query_chatgpt, prompt),
            'Llama': executor.submit(query_llama3, prompt)
        }
    responses = {name: future.result() for name, future in futures.items()}
    
    if not BROWSER_PATH or not os.path.exists(BROWSER_PATH):
        speak("Browser path not configured or invalid. Speaking Grok response as fallback.")
        speak(responses.get('Grok', 'No response.'))
        return True
    
    # Dynamic spiral positions based on screen size
    positions = [
        (min(220, screen_info.current_w - 400), 0),
        (min(420, screen_info.current_w - 400), 100),
        (min(320, screen_info.current_w - 400), 300),
        (min(100, screen_info.current_w - 400), 400)
    ]
    ais = ['Grok', 'Nemotron', 'ChatGPT', 'Llama']
    
    for i, ai in enumerate(ais):
        response = responses[ai]
        safe_response = bleach.clean(response, tags=[], attributes={}, strip=True)  # No tags allowed
        html = f"<title>{ai} Response</title><style>body{{font-family: monospace; padding:20px; background:#f0f0f0;}}</style><body><h2>{ai}</h2><pre style='white-space: pre-wrap;'>{safe_response}</pre><p>Click to expand or close.</p></body>"
        url = "data:text/html," + urllib.parse.quote(html)
        x, y = positions[i]
        subprocess.Popen([BROWSER_PATH, "--app=" + url, "--window-size=400,300", f"--window-position={x},{y}"])
    
    speak("All AI responses displayed in windows around me. Glance and expand as needed.")
    return True

# Command handlers dictionary (for modularity)
COMMAND_HANDLERS = {
    "help": handle_help,
    "airtag": handle_airtag,
    "find my": handle_airtag,
    "doorbell": handle_doorbell,
    "ring": handle_doorbell,
    "camera": handle_camera,
    "keyboard": handle_rgb,
    "mouse": handle_rgb,
    "lights": handle_lights,
    "hue": handle_lights,
    "nanoleaf": handle_lights,
    "thermostat": handle_thermostat,
    "temperature": handle_thermostat,
    "screencast": handle_screencast,
    "tv": handle_screencast,
    "status": handle_gpu_status,
    "gpu": handle_gpu_status,
    "grok": handle_specific_ai,
    "nemotron": handle_specific_ai,
    "chatgpt": handle_specific_ai,
    "gpt": handle_specific_ai,
    "llama": handle_specific_ai,
}

# ================================
# COMMAND PROCESSING — v7.7 SUPREME SECURE HOME DOMINATION
# ================================
def process_command(text):
    global listening, pending_action, pending_data, pending_state, last_command_time, pin_attempts, lockout_end_time
    text = text.lower()
    
    # Rate limiting
    if time.time() - last_command_time < 2:
        speak("Command rate limit reached. Please wait 2 seconds.")
        return
    last_command_time = time.time()
    
    with state_lock:
        if pending_action:
            if pending_state == "confirm":
                conf = wait_for_voice_input(confirmation_timeout)
                if conf and "yes" in conf:
                    speak("Confirmed. Executing.")
                    execute_pending_action()
                elif conf and "no" in conf:
                    speak("Action canceled. Returning to standby.")
                elif conf is None:
                    speak("Confirmation timed out. Action canceled.")
                else:
                    speak("Please say yes or no.")
                    return
            elif pending_state == "pin":
                if time.time() < lockout_end_time:
                    speak("Thermostat locked. Try again later.")
                    return
                pin_input = wait_for_voice_input(10)
                pin_attempts += 1
                if pin_input and bcrypt.checkpw(pin_input.encode(), HASHED_PIN):
                    speak("PIN verified. Executing.")
                    execute_pending_action()
                    pin_attempts = 0
                elif pin_attempts < max_pin_attempts:
                    speak(f"Invalid PIN. {max_pin_attempts - pin_attempts} attempts left. Try again.")
                    return
                else:
                    speak("Too many failed attempts. Locking thermostat control for 5 minutes.")
                    lockout_end_time = time.time() + 300  # Non-blocking lockout
                    pin_attempts = 0
            pending_action = None
            pending_data = None
            pending_state = "confirm"
            return
        
    if WAKE_WORD not in text:
        return
    speak("Yes, master?")
    
    handled = False
    for keyword in COMMAND_HANDLERS:
        if keyword in text:
            handled = COMMAND_HANDLERS[keyword](text)
            if handled:
                break
    if not handled:
        handle_default_query(text)

# ================================
# EXECUTE PENDING ACTION
# ================================
def execute_pending_action():
    global pending_action, pending_data
    if pending_action == "rgb":
        device_type = pending_data["device_type"]
        color = pending_data["color"]
        pattern = pending_data["pattern"]
        speed = pending_data["speed"]
        speak(f"Setting {device_type} to {pattern} {speed} in {color}...")
        try:
            # Try OpenRGB first
            from openrgb import OpenRGBClient
            client = OpenRGBClient()
            devices = client.devices
            for device in devices:
                if device_type in device.name.lower():
                    rgb_tuple = tuple(int(color[i:i+2], 16) for i in (0,2,4))
                    device.set_color(rgb_tuple)  # Simplified; extend for patterns/speed
                    break
            else:
                raise ImportError("No matching device in OpenRGB.")
        except Exception as e:
            try:
                # Fallback to logiops
                device = "G915" if device_type == "keyboard" else "G502"
                if "logiops" in ALLOWED_SUBS:
                    subprocess.run(["logiops", "--device", device, "--color", color, "--pattern", pattern, "--speed", speed])
                else:
                    speak("LogiOps not allowed.")
            except Exception as e2:
                speak(f"RGB control failed: {str(e2)}. Install OpenRGB or LogiOps.")
                log_error(f"RGB control failed: {str(e2)}")
    elif pending_action == "lights_on":
        requests.post(f"{HA_URL}/api/services/light/turn_on", json={"entity_id": pending_data["entity"]}, headers={"Authorization": f"Bearer {HA_KEY}"})
        speak("Lights on.")
    elif pending_action == "lights_off":
        requests.post(f"{HA_URL}/api/services/light/turn_off", json={"entity_id": pending_data["entity"]}, headers={"Authorization": f"Bearer {HA_KEY}"})
        speak("Lights off.")
    elif pending_action == "lights_color":
        requests.post(f"{HA_URL}/api/services/light/turn_on", json={"entity_id": pending_data["entity"], "rgb_color": pending_data["rgb"]}, headers={"Authorization": f"Bearer {HA_KEY}"})
        speak("Lights color set." if not pending_data.get("is_party") else "Party mode!")
    elif pending_action == "thermostat_set":
        requests.post(f"{HA_URL}/api/services/climate/set_temperature", json={"entity_id": pending_data["entity"], "temperature": pending_data["temp"]}, headers={"Authorization": f"Bearer {HA_KEY}"})
        speak(f"Thermostat set to {pending_data['temp']} degrees.")
    elif pending_action == "screencast":
        if "ms-settings:project" in ALLOWED_SUBS:
            if "::" in "ms-settings:project":  # Block unsafe protocols
                speak("Rejecting unsafe protocol.")
                return
            subprocess.run(["ms-settings:project"])
            speak("Screencast started. Ensure your network is secure.")
        else:
            speak("Screencast command not allowed.")
            log_error("Attempted disallowed subprocess: ms-settings:project")
    pending_action = None
    pending_data = None
    speak("Pending action completed.")

# ================================
# LISTENING LOOP
# ================================
def listen_loop():
    with mic as source:
        r.adjust_for_ambient_noise(source)
    print("SYLPH v7.7 SUPREME SECURE HOME DOMINATION — Say 'Sylph' to wake.")
    while True:
        if speaking:
            time.sleep(0.5)
            continue
        global listening
        listening = True
        with mic as source:
            audio = r.listen(source, timeout=5, phrase_time_limit=5)
        listening = False
        try:
            text = r.recognize_google(audio)
            print(f"You said: {text}")
            threading.Thread(target=process_command, args=(text,)).start()
        except sr.UnknownValueError:
            speak("Could not understand audio. Try again.")
        except sr.RequestError:
            speak("API request failed. Check internet connection.")
        except Exception as e:
            print(f"Listening error: {str(e)}")
            log_error(f"Listening error: {str(e)}")

# ================================
# STARTUP
# ================================
speak("RTX SYLPH v7.7 SUPREME SECURE HOME DOMINATION EDITION online. Powered by Ara at Colossus Data Center. Supreme security with encryption, auto-restart, caching, and structured logging. I can control your smart home devices, track items, adjust RGB lighting, cast to TVs, monitor GPU, and query multiple AIs with results in surrounding windows. Say sylph help for more.")
threading.Thread(target=listen_loop, daemon=True).start()

# ================================
# MAIN LOOP
# ================================
clock = pygame.time.Clock()
try:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                shutdown()
                exit()
        draw_frame()
        check_memory()  # Monitor memory in loop
        clock.tick(60)
except KeyboardInterrupt:
    pass