"""
Локальный голосовой агент "Добрый Альтрон"
Пайплайн: микрофон -> STT (faster-whisper) -> LLM (Qwen через Ollama)
          -> TTS (XTTS v2) -> пост-обработка звука -> динамики

Рассчитано на RTX 3060 Laptop (6 ГБ VRAM), Windows 10.
Whisper работает на CPU, чтобы не конкурировать за VRAM с LLM и TTS.
"""

import io
import os
import time
import wave
import numpy as np
import torch
import sounddevice as sd
import requests
from faster_whisper import WhisperModel
from TTS.api import TTS
from pedalboard import Pedalboard, PitchShift, Distortion, Reverb, HighpassFilter
from pedalboard.io import AudioFile

# ---------------------------------------------------------------------------
# Настройки
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:7b-instruct-q4_K_M"  # если не влезает по VRAM -> qwen2.5:3b-instruct

WHISPER_MODEL_SIZE = "small"   # base/small/medium; на CPU small — разумный баланс
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"  # быстрее на CPU

XTTS_LOCAL_DIR = "xtts_v2_local"  # скачивается скриптом download_model.py
REFERENCE_VOICE = "reference_voice.wav"  # твой референсный сэмпл голоса (нейтральный мужской, глубокий)
VOICE_CACHE_PATH = "altron_voice.pt"     # сюда сохраняем посчитанный один раз эмбеддинг голоса
LANGUAGE = "ru"

SAMPLE_RATE = 16000
RECORD_SECONDS = 6  # длительность записи одной реплики пользователя

SYSTEM_PROMPT = (
    "Ты — Альтрон, но добрая версия: рассудительный, спокойный, слегка ироничный "
    "искусственный интеллект, который заботится о людях, а не хочет их уничтожить. "
    "Отвечай кратко и по делу, без длинных вступлений, на русском языке."
)

# ---------------------------------------------------------------------------
# Инициализация моделей
# ---------------------------------------------------------------------------

print("[*] Загружаю Whisper (STT)...")
stt_model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)

print("[*] Загружаю XTTS v2 (TTS)...")
tts_model = TTS(
    model_path=XTTS_LOCAL_DIR,
    config_path=os.path.join(XTTS_LOCAL_DIR, "config.json"),
)
tts_model.to("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Кэш голоса Альтрона: считаем эмбеддинг (latents) один раз и сохраняем на диск,
# дальше просто загружаем его, не трогая reference_voice.wav повторно.
# ---------------------------------------------------------------------------

def get_or_create_voice_latents():
    if os.path.exists(VOICE_CACHE_PATH):
        print(f"[*] Нашёл сохранённый голос Альтрона: {VOICE_CACHE_PATH}")
        data = torch.load(VOICE_CACHE_PATH)
        return data["gpt_cond_latent"], data["speaker_embedding"]

    print("[*] Сохранённого голоса нет, считаю эмбеддинг из reference_voice.wav "
          "(это нужно будет сделать только один раз)...")
    gpt_cond_latent, speaker_embedding = tts_model.synthesizer.tts_model.get_conditioning_latents(
        audio_path=REFERENCE_VOICE
    )
    torch.save(
        {"gpt_cond_latent": gpt_cond_latent, "speaker_embedding": speaker_embedding},
        VOICE_CACHE_PATH,
    )
    print(f"[*] Голос сохранён в {VOICE_CACHE_PATH}. В следующий раз пересчёт не понадобится.")
    return gpt_cond_latent, speaker_embedding


# ---------------------------------------------------------------------------
# Запись голоса пользователя
# ---------------------------------------------------------------------------

def record_audio(seconds=RECORD_SECONDS, samplerate=SAMPLE_RATE):
    print(f"[*] Говори! Запись {seconds} сек...")
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


# ---------------------------------------------------------------------------
# STT
# ---------------------------------------------------------------------------

def transcribe(audio: np.ndarray) -> str:
    segments, _ = stt_model.transcribe(audio, language="ru", beam_size=5)
    text = " ".join(seg.text.strip() for seg in segments)
    print(f"[STT] {text}")
    return text


# ---------------------------------------------------------------------------
# LLM через Ollama
# ---------------------------------------------------------------------------

def ask_llm(user_text: str, history: list) -> str:
    history.append({"role": "user", "content": user_text})
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + history,
        "stream": False,
        # keep_alive=0 выгружает модель из VRAM сразу после ответа,
        # освобождая память для TTS. Убери, если VRAM хватает на всё сразу.
        "keep_alive": "0s",
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    answer = resp.json()["message"]["content"]
    history.append({"role": "assistant", "content": answer})
    print(f"[LLM] {answer}")
    return answer


# ---------------------------------------------------------------------------
# TTS + пост-обработка "под Альтрона"
# ---------------------------------------------------------------------------

def synthesize(text: str, gpt_cond_latent, speaker_embedding, out_path: str = "reply_raw.wav"):
    # Используем уже посчитанные latents вместо speaker_wav —
    # так XTTS не пересчитывает голос заново на каждой фразе/запуске.
    out = tts_model.synthesizer.tts_model.inference(
        text=text,
        language=LANGUAGE,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
    )
    wav = np.array(out["wav"])

    with AudioFile(out_path, "w", 24000, 1) as f:
        f.write(wav.reshape(1, -1))

    return out_path


def altronize_voice(in_path: str, out_path: str = "reply_altron.wav"):
    """
    Накладываем эффекты: понижение питча, лёгкое искажение,
    металлический реверб, срез низких частот для "механического" тембра.
    Никакого клонирования конкретного актёра — только обработка тембра.
    """
    board = Pedalboard([
        PitchShift(semitones=-3),
        HighpassFilter(cutoff_frequency_hz=120),
        Distortion(drive_db=6),
        Reverb(room_size=0.3, damping=0.6, wet_level=0.15, dry_level=0.85),
    ])

    with AudioFile(in_path) as f:
        audio = f.read(f.frames)
        samplerate = f.samplerate

    processed = board(audio, samplerate)

    with AudioFile(out_path, "w", samplerate, processed.shape[0]) as f:
        f.write(processed)

    return out_path


def play_audio(path: str):
    with AudioFile(path) as f:
        audio = f.read(f.frames)
        sd.play(audio.T, f.samplerate)
        sd.wait()


# ---------------------------------------------------------------------------
# Главный цикл
# ---------------------------------------------------------------------------

def main():
    history = []
    gpt_cond_latent, speaker_embedding = get_or_create_voice_latents()

    print("=== Добрый Альтрон запущен. Ctrl+C для выхода. ===")
    try:
        while True:
            input("\nНажми Enter и начни говорить...")
            audio = record_audio()
            user_text = transcribe(audio)

            if not user_text.strip():
                print("[!] Ничего не расслышал, попробуй ещё раз.")
                continue

            reply_text = ask_llm(user_text, history)

            raw_path = synthesize(reply_text, gpt_cond_latent, speaker_embedding)
            final_path = altronize_voice(raw_path)

            print("[*] Воспроизвожу ответ...")
            play_audio(final_path)

    except KeyboardInterrupt:
        print("\n[*] Завершение работы.")


if __name__ == "__main__":
    main()
