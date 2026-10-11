# Ultron Agent — добрый локальный Альтрон с голосом

Локальный голосовой ассистент: микрофон → STT (faster-whisper) → LLM (Qwen через Ollama) → TTS (XTTS v2) → обработка под «металлический» голос (pedalboard).

Проверено на: Windows 10, RTX 3060 Laptop (6 ГБ VRAM), Python 3.11.

## Состав

| Файл | Назначение |
|---|---|
| `download_model.py` | Скачивает XTTS v2 с Hugging Face в `xtts_v2_local/` (с повторами при обрывах) |
| `convert_audio.py` | Делает `reference_voice.wav` из mp3/mp4/m4a и др. (ffmpeg встроен через `imageio-ffmpeg`) |
| `save_voice.py` | Один раз считает голос из `reference_voice.wav` и сохраняет в `altron_voice.pt` |
| `test_voice.py` | Разовая проверка: озвучивает фразу и сохраняет `test_raw.wav` / `test_altron.wav` |
| `tts_server.py` | Локальный сервер озвучки: модель грузится один раз и остаётся в памяти |
| `speak.py` | Клиент сервера: `python speak.py "текст"` |
| `voice_agent.py` | Полный цикл голосового диалога (STT → LLM → TTS) |

## Установка

### 1. Python 3.11
Пакет `TTS` (Coqui) работает только с Python 3.9–3.11. На 3.12+ он не ставится.

### 2. Microsoft C++ Build Tools
Нужны, чтобы собрался один C-модуль внутри `TTS`. В Visual Studio Installer отметь рабочую нагрузку **«Разработка классических приложений на C++»** вместе с **Windows 10/11 SDK**. Без SDK сборка падает с ошибкой `io.h: No such file or directory`.

### 3. Окружение и зависимости
```
py -3.11 -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
PyTorch ставится отдельной командой, а не через `requirements.txt`: так pip не путает индексы пакетов.

Проверка CUDA:
```
python -c "import torch; print(torch.cuda.is_available())"
```

### 4. Ollama и Qwen
Установи Ollama (https://ollama.com/download), затем:
```
ollama pull qwen2.5:7b-instruct-q4_K_M
```
Если не хватает VRAM, возьми `qwen2.5:3b-instruct` и поменяй `OLLAMA_MODEL` в `voice_agent.py`.

### 5. Модель XTTS
```
python download_model.py
```
В конце скрипт печатает размеры файлов. `model.pth` должен быть порядка гигабайта и больше, `speakers_xtts.pth` — не несколько КБ.

### 6. Референсный голос
Нужна запись 10–30 секунд чистой речи одного человека без музыки и шума. Если исходник не WAV:
```
python convert_audio.py my_voice.mp4
```
Получится `reference_voice.wav`. Не используй записи голоса реальных актёров озвучки персонажей: это может нарушать право на голос.

```
python save_voice.py
```

## Использование

Быстрая проверка голоса:
```
python test_voice.py "Как делишки у моей малышки?"
```

Мгновенная озвучка (модель держится в памяти):
```
python tts_server.py          # окно 1, оставить открытым
python speak.py "Привет"      # окно 2
```

Голосовой диалог целиком (нужен запущенный Ollama):
```
python voice_agent.py
```

## Настройка звучания
Константы в начале `test_voice.py` и `tts_server.py`: `PITCH_SEMITONES` (ниже число — ниже голос), `DISTORTION_DB` (металличность), `REVERB_WET` (эхо).

## Про VRAM (6 ГБ)
Qwen 7B (q4) и XTTS одновременно впритык не помещаются, поэтому в `voice_agent.py` Whisper работает на CPU, а Ollama выгружает модель после каждого ответа (`keep_alive: "0s"`). При нехватке памяти используй модель 3B.

## Не попадает в git (см. `.gitignore`)
`venv/`, `xtts_v2_local/` (несколько ГБ), `reference_voice.wav`, `altron_voice.pt`, исходные записи голоса и тестовые wav.

## Планы
- переписать `voice_agent.py` на обращение к `tts_server.py`;
- VAD (silero-vad) вместо записи по таймеру;
- function calling, память (Chroma), инструменты — превращение в полноценного агента.
