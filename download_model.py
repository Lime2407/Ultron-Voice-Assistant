"""
Разовая загрузка модели XTTS v2 с Hugging Face в локальную папку xtts_v2_local.
Запусти один раз:
    python download_model.py

Скачивание можно прерывать и запускать снова — докачает недостающее.
"""

import os

# Отключаем Xet-бэкенд: на некоторых сетях он падает с "error decoding response body".
# Переменная должна быть выставлена ДО импорта huggingface_hub.
os.environ["HF_HUB_DISABLE_XET"] = "1"

import time
from huggingface_hub import snapshot_download

LOCAL_DIR = "xtts_v2_local"
FILES = ["config.json", "model.pth", "vocab.json", "speakers_xtts.pth", "dvae.pth", "mel_stats.pth"]
MAX_ATTEMPTS = 5

for attempt in range(1, MAX_ATTEMPTS + 1):
    try:
        print(f"[*] Попытка {attempt}/{MAX_ATTEMPTS}: скачиваю coqui/XTTS-v2 (несколько ГБ)...")
        snapshot_download(
            repo_id="coqui/XTTS-v2",
            local_dir=LOCAL_DIR,
            allow_patterns=FILES,
        )
        break
    except Exception as e:
        print(f"[!] Ошибка: {e}")
        if attempt == MAX_ATTEMPTS:
            raise
        print("[*] Повторю через 10 секунд, уже скачанное не потеряется...")
        time.sleep(10)

print(f"\n[+] Готово. Модель лежит в папке {LOCAL_DIR}. Размеры файлов:")
for name in FILES:
    path = os.path.join(LOCAL_DIR, name)
    if os.path.exists(path):
        print(f"    {name:20s} {os.path.getsize(path) / 1024 / 1024:10.2f} МБ")
    else:
        print(f"    {name:20s} ОТСУТСТВУЕТ")
