"""
Локальный сервер озвучки: XTTS v2 и голос Альтрона загружаются ОДИН РАЗ
при старте и остаются в памяти. Дальше каждый запрос на озвучку
выполняется почти мгновенно, без повторной загрузки модели.

Запуск (оставь окно открытым, пока пользуешься голосом):
    python tts_server.py

Проверка из другого окна терминала:
    python speak.py "Как делишки у моей малышки?"

voice_agent.py тоже можно переключить на обращение к этому серверу
вместо своей собственной загрузки XTTS — тогда общий цикл STT->LLM->TTS
не будет тратить время на инициализацию модели при каждом запуске.
"""

import io
import os
import sys

import numpy as np
import torch
from flask import Flask, request, send_file
from pedalboard import Distortion, HighpassFilter, Pedalboard, PitchShift, Reverb
from pedalboard.io import AudioFile
from TTS.api import TTS

XTTS_LOCAL_DIR = "xtts_v2_local"
VOICE_CACHE_PATH = "altron_voice.pt"
LANGUAGE = "ru"
HOST = "127.0.0.1"
PORT = 5005

PITCH_SEMITONES = -3
DISTORTION_DB = 6
REVERB_WET = 0.15

XTTS_SAMPLE_RATE = 24000

for path, hint in [(XTTS_LOCAL_DIR, "python download_model.py"), (VOICE_CACHE_PATH, "python save_voice.py")]:
    if not os.path.exists(path):
        print(f"[!] Не найден {path}. Сначала запусти: {hint}")
        sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[*] Устройство: {device}")
print("[*] Загружаю XTTS v2 (один раз, это может занять до минуты)...")

tts_model = TTS(
    model_path=XTTS_LOCAL_DIR,
    config_path=os.path.join(XTTS_LOCAL_DIR, "config.json"),
)
tts_model.to(device)
xtts = tts_model.synthesizer.tts_model
xtts.to(device)

print(f"[*] Загружаю голос из {VOICE_CACHE_PATH}...")
voice_data = torch.load(VOICE_CACHE_PATH, map_location=device)
gpt_cond_latent = voice_data["gpt_cond_latent"]
speaker_embedding = voice_data["speaker_embedding"]

board = Pedalboard([
    PitchShift(semitones=PITCH_SEMITONES),
    HighpassFilter(cutoff_frequency_hz=120),
    Distortion(drive_db=DISTORTION_DB),
    Reverb(room_size=0.3, damping=0.6, wet_level=REVERB_WET, dry_level=1.0 - REVERB_WET),
])

app = Flask(__name__)
print(f"[+] Готово. Сервер слушает на http://{HOST}:{PORT}")
print("    Держи это окно открытым, пока хочешь мгновенную озвучку.")


@app.route("/health")
def health():
    return {"status": "ok", "device": device}


@app.route("/speak", methods=["POST"])
def speak():
    payload = request.get_json(force=True)
    text = (payload or {}).get("text", "").strip()
    if not text:
        return {"error": "поле 'text' пустое"}, 400

    raw = payload.get("raw", False)  # raw=true -> без эффектов, для отладки

    out = xtts.inference(
        text=text,
        language=LANGUAGE,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
    )
    wav = np.asarray(out["wav"], dtype=np.float32)

    if raw:
        final = wav.reshape(1, -1)
    else:
        padded = np.pad(wav, (0, int(XTTS_SAMPLE_RATE * 0.6))).reshape(1, -1)
        final = board(padded, XTTS_SAMPLE_RATE)

    buf = io.BytesIO()
    with AudioFile(buf, "w", XTTS_SAMPLE_RATE, 1, format="wav") as f:
        f.write(final)
    buf.seek(0)

    return send_file(buf, mimetype="audio/wav")


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=False)
