"""
Тест голоса Альтрона: произносит фразу без Ollama и микрофона.

Запуск:
    python test_voice.py
    python test_voice.py "Любая другая фраза для проверки"

Нужно, чтобы уже были:
    - папка xtts_v2_local (скрипт download_model.py)
    - файл altron_voice.pt (скрипт save_voice.py)

На выходе два файла, чтобы можно было сравнить:
    test_raw.wav    - голос XTTS без эффектов
    test_altron.wav - тот же голос с "альтроновской" обработкой
"""

import os
import sys

import numpy as np
import sounddevice as sd
import torch
from pedalboard import Distortion, HighpassFilter, Pedalboard, PitchShift, Reverb
from pedalboard.io import AudioFile
from TTS.api import TTS

# ---------------------------------------------------------------------------
# Настройки (крути их, чтобы подобрать звучание)
# ---------------------------------------------------------------------------

PHRASE = "Как делишки у моей малышки?"

XTTS_LOCAL_DIR = "xtts_v2_local"
VOICE_CACHE_PATH = "altron_voice.pt"
LANGUAGE = "ru"

PITCH_SEMITONES = -3     # ниже число -> голос ниже (например -5 для более "тяжёлого")
DISTORTION_DB = 6        # 0 -> без "металла", 10+ -> заметно "грязнее"
REVERB_WET = 0.15        # 0 -> без эха

RAW_PATH = "test_raw.wav"
ALTRON_PATH = "test_altron.wav"
XTTS_SAMPLE_RATE = 24000

# ---------------------------------------------------------------------------

if len(sys.argv) > 1:
    PHRASE = " ".join(sys.argv[1:])

for path, hint in [(XTTS_LOCAL_DIR, "python download_model.py"), (VOICE_CACHE_PATH, "python save_voice.py")]:
    if not os.path.exists(path):
        print(f"[!] Не найден {path}. Сначала запусти: {hint}")
        sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Устройство: {device}")

print("[*] Загружаю XTTS v2...")
tts_model = TTS(
    model_path=XTTS_LOCAL_DIR,
    config_path=os.path.join(XTTS_LOCAL_DIR, "config.json"),
)
tts_model.to(device)
xtts = tts_model.synthesizer.tts_model
xtts.to(device)

print(f"[*] Загружаю сохранённый голос из {VOICE_CACHE_PATH}...")
data = torch.load(VOICE_CACHE_PATH, map_location=device)
gpt_cond_latent = data["gpt_cond_latent"]
speaker_embedding = data["speaker_embedding"]

print(f"[*] Синтезирую: {PHRASE}")
out = xtts.inference(
    text=PHRASE,
    language=LANGUAGE,
    gpt_cond_latent=gpt_cond_latent,
    speaker_embedding=speaker_embedding,
)
wav = np.asarray(out["wav"], dtype=np.float32)

with AudioFile(RAW_PATH, "w", XTTS_SAMPLE_RATE, 1) as f:
    f.write(wav.reshape(1, -1))
print(f"[+] Сохранил {RAW_PATH} (без эффектов)")

# Немного тишины в конце, чтобы хвост реверберации не обрезался
padded = np.pad(wav, (0, int(XTTS_SAMPLE_RATE * 0.6))).reshape(1, -1)

board = Pedalboard([
    PitchShift(semitones=PITCH_SEMITONES),
    HighpassFilter(cutoff_frequency_hz=120),
    Distortion(drive_db=DISTORTION_DB),
    Reverb(room_size=0.3, damping=0.6, wet_level=REVERB_WET, dry_level=1.0 - REVERB_WET),
])
processed = board(padded, XTTS_SAMPLE_RATE)

with AudioFile(ALTRON_PATH, "w", XTTS_SAMPLE_RATE, 1) as f:
    f.write(processed)
print(f"[+] Сохранил {ALTRON_PATH} (с эффектами Альтрона)")

try:
    print("[*] Проигрываю версию с эффектами...")
    sd.play(processed.T, XTTS_SAMPLE_RATE)
    sd.wait()
except Exception as e:
    print(f"[!] Не удалось проиграть звук ({e}). Открой {ALTRON_PATH} вручную.")
