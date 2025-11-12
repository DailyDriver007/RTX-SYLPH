# sylph.py - RTX Sylph v6.0 - COLOSSUS EDITION: Full G-Assist Plugin + config.json + Lip Sync
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
# LOAD CONFIG FROM config.json
# ================================
CONFIG_PATH = Path(__file__).parent / "config.json"
if not CONFIG_PATH.exists():
    raise FileNotFoundError("config.json missing! Create it in the G-Assist folder with your API keys.")

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

# Required
VIDEO_PATH = "assets/rtx_sylph_animated.mp4"
WAKE_WORD = config.get("WAKE_WORD", "sylph").lower()
VOICE_SPEED = config.get("VOICE_SPEED", 155)

# API Keys (with fallbacks)
GROK_API_KEY = config.get("GROK_API_KEY", "").strip()
OPENAI_API_KEY = config.get("OPENAI_API_KEY", "").strip()
NVIDIA_API_KEY = config.get("NVIDIA_API_KEY", "").strip()
HUGGINGFACE_API_KEY = config.get("HUGGINGFACE_API_KEY", "").strip()

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
    raise FileNotFoundError(f"Video not found at {VIDEO_PATH}")

engine = pyttsx3.init()
engine.setProperty('rate', VOICE_SPEED)
engine.setProperty('voice', 'zira')

r = sr.Recognizer()
mic = sr.Microphone()

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
# SPEAK WITH STATE
# ================================
def speak(text):
 global speaking, listening
    speaking = True
    listening = False
    engine.say(text)
    engine.runAndWait()
    speaking = False

# ================================
# LLM CONNECTORS
# ================================
def query_grok(prompt):
    if not GROK_API_KEY: return "Grok API key missing in config.json"
    headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
    data = {"model": "grok-beta", "messages": [{"role": "user", "content": prompt}], "temperature": 0.7}
    try:
        resp = requests.post(GROK_URL, headers=headers, json=data, timeout=20)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except: return "Grok offline or bad key."

def query_chatgpt(prompt):
    if not OPENAI_API_KEY: return "ChatGPT key missing"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    data = {"model": "gpt-4o", "messages": [{"role": "user", "content": prompt}]}
    try:
        resp = requests.post(OPENAI_URL, headers=headers, json=data, timeout=20)
        return resp.json()["choices"][0]["message"]["content"]
    except: return "ChatGPT offline."

def query_nemotron(prompt):
    if not NVIDIA_API_KEY: return "Nemotron key missing"
    headers = {"Authorization": f"Bearer {NVIDIA_API_KEY}", "Content-Type": "application/json"}
    data = {"model": "nvidia/nemotron-4-340b-reward", "messages": [{"role": "user", "content": prompt}]}
    try:
        resp = requests.post(NVIDIA_NEMOTRON_URL, headers=headers, json=data, timeout=20)
        return resp.json()["choices"][0]["message"]["content"]
    except: return "Nemotron offline."

def query_llama3(prompt):
    if not HUGGINGFACE_API_KEY: return "Llama-3 key missing"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}", "Content-Type": "application/json"}
    data = {"inputs": f"[INST] {prompt} [/INST]", "parameters": {"max_new_tokens": 150}}
    try:
        resp = requests.post(HF_LLAMA_URL, headers=headers, json=data, timeout=20)
        return resp.json()[0]["generated_text"].split("[/INST]")[-1].strip()
    except: return "Llama-3 offline."

# ================================
# COMMAND PROCESSING
# ================================
def process_command(text):
    global listening
    text = text.lower()
    if WAKE_WORD not in text:
        return
    speak("Yes, master?")
    if "status" in text or "gpu" in text:
        gpu = GPUtil.getGPUs()[0]
        msg = f"RTX {gpu.name.split()[-1]}. Load {gpu.load*100:.0f}%. RAM {gpu.memoryUsed}MB. Temp {gpu.temperature}°C."
        speak(msg)
    elif "grok" in text:
        speak("Asking Grok...")
        response = query_grok(text)
        speak(response)
    elif "nemotron" in text:
        speak("Asking Nemotron...")
        response = query_nemotron(text)
        speak(response)
    elif "chatgpt" in text or "gpt" in text:
        speak("Asking ChatGPT...")
        response = query_chatgpt(text)
        speak(response)
    elif "llama" in text:
        speak("Asking Llama-3...")
        response = query_llama3(text)
        speak(response)
    else:
        speak("Try: status, grok, nemotron, chatgpt, llama.")

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
        with mic as source:
            audio = r.listen(source, timeout=5, phrase_time_limit=5)
        listening = False
        try:
            text = r.recognize_google(audio)
            print(f"You said: {text}")
            threading.Thread(target=process_command, args=(text,)).start()
        except:
            pass

# ================================
# STARTUP
# ================================
speak("RTX Sylph online. Colossus mode activated.")
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
    clock.tick(60)