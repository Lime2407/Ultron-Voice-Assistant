"""
Достаёт звук из любого аудио/видеофайла (mp4, mp3, m4a, ogg, "битый" wav...)
и сохраняет корректный reference_voice.wav для XTTS: моно, 22050 Гц, до 30 сек.

Нужен пакет imageio-ffmpeg (в нём лежит готовый ffmpeg.exe, ставить ничего
в систему не надо):
    pip install imageio-ffmpeg

Использование:
    python convert_audio.py my_voice.mp4
    python convert_audio.py my_voice.mp4 5     # начать с 5-й секунды
"""

import os
import subprocess
import sys

import imageio_ffmpeg

DST = "reference_voice.wav"

if len(sys.argv) < 2:
    print("Использование: python convert_audio.py <исходный_файл> [старт_в_секундах]")
    sys.exit(1)

src = sys.argv[1]
start = sys.argv[2] if len(sys.argv) > 2 else None

if not os.path.exists(src):
    print(f"[!] Файл не найден: {src}")
    sys.exit(1)

if os.path.abspath(src) == os.path.abspath(DST):
    print(f"[!] Исходный файл не должен называться {DST}. Переименуй его и запусти снова.")
    sys.exit(1)

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
cmd = [ffmpeg, "-y"]
if start:
    cmd += ["-ss", start]
cmd += ["-i", src, "-vn", "-ar", "22050", "-ac", "1", "-t", "30", DST]

print(f"[*] Извлекаю звук из {src}...")
result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")

if result.returncode != 0:
    print("[!] ffmpeg завершился с ошибкой. Последние строки его вывода:")
    print("\n".join(result.stderr.strip().splitlines()[-15:]))
    sys.exit(1)

size_kb = os.path.getsize(DST) / 1024
print(f"[+] Готово: {DST} ({size_kb:.0f} КБ, 22050 Гц, моно, до 30 сек)")
if size_kb < 100:
    print("[!] Файл очень маленький — возможно, в исходнике мало звука или он тихий.")
