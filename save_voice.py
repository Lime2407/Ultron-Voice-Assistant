"""
Разовый скрипт: считает голос Альтрона из reference_voice.wav
и сохраняет его в altron_voice.pt. После этого voice_agent.py
всегда будет брать голос из кэша, не трогая исходный wav.

Запусти один раз:
    python save_voice.py
"""

import os
import torch
from TTS.api import TTS

XTTS_LOCAL_DIR = "xtts_v2_local"  # скачивается скриптом download_model.py
REFERENCE_VOICE = "reference_voice.wav"
VOICE_CACHE_PATH = "altron_voice.pt"

print("[*] Загружаю XTTS v2...")
tts_model = TTS(
    model_path=XTTS_LOCAL_DIR,
    config_path=os.path.join(XTTS_LOCAL_DIR, "config.json"),
)
tts_model.to("cuda" if torch.cuda.is_available() else "cpu")

print(f"[*] Считаю голос из {REFERENCE_VOICE}...")
gpt_cond_latent, speaker_embedding = tts_model.synthesizer.tts_model.get_conditioning_latents(
    audio_path=REFERENCE_VOICE
)

torch.save(
    {"gpt_cond_latent": gpt_cond_latent, "speaker_embedding": speaker_embedding},
    VOICE_CACHE_PATH,
)

print(f"[+] Готово. Голос сохранён в {VOICE_CACHE_PATH}")
print("    Теперь reference_voice.wav больше не нужен для запуска voice_agent.py.")
