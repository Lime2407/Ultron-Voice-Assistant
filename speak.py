"""
Клиент для tts_server.py: отправляет текст на уже запущенный сервер
и мгновенно получает и проигрывает озвучку (модель заново не грузится).

Сначала в отдельном окне терминала запусти и оставь работать:
    python tts_server.py

Потом в любом другом окне:
    python speak.py "Как делишки у моей малышки?"
"""

import sys
import time

import numpy as np
import requests
import sounddevice as sd
from pedalboard.io import AudioFile
import io

SERVER_URL = "http://127.0.0.1:5005/speak"

if len(sys.argv) < 2:
    text = "Как делишки у моей малышки?"
else:
    text = " ".join(sys.argv[1:])

print(f"[*] Отправляю: {text}")
t0 = time.time()

try:
    resp = requests.post(SERVER_URL, json={"text": text}, timeout=60)
except requests.exceptions.ConnectionError:
    print("[!] Сервер не отвечает. Убедись, что в другом окне запущен: python tts_server.py")
    sys.exit(1)

if resp.status_code != 200:
    print(f"[!] Ошибка сервера: {resp.status_code} {resp.text}")
    sys.exit(1)

elapsed = time.time() - t0
print(f"[+] Получил озвучку за {elapsed:.2f} сек")

with AudioFile(io.BytesIO(resp.content)) as f:
    audio = f.read(f.frames)
    samplerate = f.samplerate

sd.play(audio.T, samplerate)
sd.wait()
