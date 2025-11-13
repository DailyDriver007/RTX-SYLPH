# sylph.py - RTX Sylph v6.2 - COLOSSUS EDITION: Full G-Assist Plugin + config.json + Lip Sync
# Changes: Non-blocking TTS, API validation/retries, better error handling, robust paths.

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
from pathlib import Path

# ================================
# LOAD CONFIG FROM config.json (Robust Path)
# ================================
# Search current dir first, then script dir as fallback
CONFIG_PATH = Path("config.json")
if not CONFIG_PATH.exists():
    CONFIG_PATH = Path(__file__).parent / "config.json"
if not CONFIG_PATH.exists():
    raise FileNotFoundError("config.json missing! Create it in the current folder with your API keys.")

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# Required
VIDEO_PATH = "assets/rtx_sylph_animated.mp4"  # .mp4 for video animation (not .png)
WAKE_WORD = config.get("WAKE_WORD", "sylph").lower()
VOICE_SPEED = config.get("VOICE_SPEED", 155)

# API Keys (with validation)
GROK_API_KEY = config.get("GROK_API_KEY", "").strip()
OPENAI_API_KEY = config.get("OPENAI_API_KEY", "").strip()
NVIDIA_API_KEY = config.get("NVIDIA_API_KEY", "").strip()
HUGGINGFACE_API_KEY = config.get("HUGGINGFACE_API_KEY", "").strip()

# Validate keys early
if not GROK_API_KEY:
    raise ValueError("Missing GROK_API_KEY in config.json")
if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in config.json")
if not NVIDIA_API_KEY:
    raise ValueError("Missing NVIDIA_API_KEY in config.json")
if not HUGGINGFACE_API_KEY:
    raise ValueError("Missing HUGGINGFACE_API_KEY in config.json")

# API URLs (never change)
GROK_URL = "https://api.x.ai/v1/chat/completions"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
NVIDIA_NEMOTRON_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
HF_LLAMA_URL = "https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct"

# ================================
# INIT
# ================================
pygame.init()
screen = pygame.display.set_mode((200, 300), pygame.NOFRAME)
pygame.display.set_caption("RTX Sylph")
pygame.mouse.set_visible(False)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Video not found at {VIDEO_PATH}. Ensure .mp4 is lightweight for performance.")

engine = pyttsx3.init()
engine.setProperty('rate', VOICE_SPEED)
engine.setProperty('voice', 'zira')  # Windows TTS voice

r = sr.Recognizer()
try:
    mic = sr.Microphone()  # Add device_index=N if needed for Windows permissions
except Exception as e:
    raise RuntimeError(f"Microphone init failed: {e}. Check Windows privacy settings.")

listening = False
speaking = False

# ================================
# VIDEO LOOP WITH STATE OVERLAY
# ================================
def draw_frame():
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Video failed to load frames.")  # Prevent freeze
    if ret:
        frame = cv2.resize(frame, (200, 300))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.rot90(frame)
        surf = pygame.surfarray.make_surface(frame)
        
        # LISTENING GLOW (blue pulse)
        if listening and not speaking:
            overlay = pygame.Surface((200, 300), pygame.SRCALPHA)
            overlay.fill((0, 100, 255, 80))
            surf.blit(overlay, (0, 0))
        
        # SPEAKING GLOW (green pulse + mouth highlight)
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
# NON-BLOCKING SPEAK
# ================================
def speak_async(text):
    global speaking
    speaking = True
    try:
        engine.say(text)
        engine.runAndWait()
    finally:
        speaking = False

# ================================
# LLM QUERY WRAPPERS (with retries and error handling)
# ================================
def query_with_retry(url, headers, data, max_retries=3):
    retries = 0
    while retries < max_retries:
        try:
            resp = requests.post(url, headers=headers, json=data, timeout=20)
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as e:
            print(f"API error: {e}. Retrying...")
            retries += 1
            time.sleep(5)
    return None

def safe_query(model_func, prompt):
    try:
        return model_func(prompt)
    except Exception as e:
        print(f"LLM error: {str(e)}")
        return "Failed to process request due to error."

def query_grok(prompt):
    headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
    data = {"model": "grok-beta", "messages": [{"role": "user", "content": prompt}], "temperature": 0.7}
    resp = query_with_retry(GROK_URL, headers, data)
    return resp.json()["choices"][0]["message"]["content"] if resp else "Grok offline or bad key."

def query_chatgpt(prompt):
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    data = {"model": "gpt-4o", "messages": [{"role": "user", "content": prompt}]}
    resp = query_with_retry(OPENAI_URL, headers, data)
    return resp.json()["choices"][0]["message"]["content"] if resp else "ChatGPT offline."

def query_nemotron(prompt):
    headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"}
    data = {"model": "nvidia/nemotron-4-340b-reward", "messages": [{"role": "user", "content": prompt}]}
    resp = query_with_retry(NVIDIA_NEMOTRON_URL, headers, data)
    return resp.json()["choices"][0]["message"]["content"] if resp else "Nemotron offline."

def query_llama3(prompt):
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}", "Content-Type": "application/json"}
    data = {"inputs": f"[INST] {prompt} [/INST]", "parameters": {"max_new_tokens": 150}}
    resp = query_with_retry(HF_LLAMA_URL, headers, data)
    return resp.json()[0]["generated_text"].split("[/INST]")[-1].strip() if resp else "Llama-3 offline."

# ================================
# COMMAND PROCESSING
# ================================
def process_command(text):
    global listening
    text = text.lower()
    if WAKE_WORD not in text:
        return
    threading.Thread(target=speak_async, args=("Yes, master?",)).start()
    if "status" in text or "gpu" in text:
        gpu = GPUtil.getGPUs()[0]
        msg = f"RTX {gpu.name.split()[-1]}. Load {gpu.load*100:.0f}%. RAM {gpu.memoryUsed}MB. Temp {gpu.temperature}°C."
        threading.Thread(target=speak_async, args=(msg,)).start()
    elif "grok" in text:
        threading.Thread(target=speak_async, args=("Asking Grok...",)).start()
        response = safe_query(query_grok, text)
        threading.Thread(target=speak_async, args=(response,)).start()
    elif "nemotron" in text:
        threading.Thread(target=speak_async, args=("Asking Nemotron...",)).start()
        response = safe_query(query_nemotron, text)
        threading.Thread(target=speak_async, args=(response,)).start()
    elif "chatgpt" in text or "gpt" in text:
        threading.Thread(target=speak_async, args=("Asking ChatGPT...",)).start()
        response = safe_query(query_chatgpt, text)
        threading.Thread(target=speak_async, args=(response,)).start()
    elif "llama" in text:
        threading.Thread(target=speak_async, args=("Asking Llama-3...",)).start()
        response = safe_query(query_llama3, text)
        threading.Thread(target=speak_async, args=(response,)).start()
    else:
        threading.Thread(target=speak_async, args=("Try: status, grok, nemotron, chatgpt, llama.",)).start()

# ================================
# LISTENING LOOP
# ================================
def listen_loop():
    with mic as source:
        r.adjust_for_ambient_noise(source)
    print("Sylph listening... Say 'Sylph' to wake.")
    while True:
        if speaking:
            time.sleep(0.5)
            continue
        global listening
        listening = True
        try:
            with mic as source:
                audio = r.listen(source, timeout=5, phrase_time_limit=5)
            text = r.recognize_google(audio)
            print(f"You said: {text}")
            threading.Thread(target=process_command, args=(text,)).start()
        except Exception as e:
            print(f"Recognition error: {e}")
        finally:
            listening = False

# ================================
# STARTUP
# ================================
threading.Thread(target=speak_async, args=("RTX Sylph online. Colossus mode activated.",)).start()
threading.Thread(target=listen_loop, daemon=True).start()

# ================================
# MAIN LOOP
# ================================
clock = pygame.time.Clock()
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            exit()
    draw_frame()
    clock.tick(60)  # Can lower to 30 for lower-end GPUs