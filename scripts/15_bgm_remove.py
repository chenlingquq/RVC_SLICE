import subprocess
import sys
from pathlib import Path
import csv
import json
import shutil

# ========= 路径配置 =========
ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "work" / "ingested"
OUT_DIR = ROOT / "work" / "denoise"
MANIFEST_DIR = ROOT / "work" / "manifests"

OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = MANIFEST_DIR / "demucs_manifest.csv"
JSONL_PATH = MANIFEST_DIR / "demucs_manifest.jsonl"

# ========= Demucs 参数 =========
DEMUCS_MODEL = "htdemucs"   # 默认就好，后续你可以换 mdx_extra_q
DEVICE = "cuda"             # 你已经确认 CUDA OK
JOBS = "1"                  # Windows 下 1 最稳

# ========= 收集音频 =========
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a"}
audio_files = [p for p in INPUT_DIR.iterdir() if p.suffix.lower() in AUDIO_EXTS]

results = []

print(f"[INFO] Found {len(audio_files)} audio files.")

for audio in audio_files:
    print(f"[DEMUX] {audio.name}")

    cmd = [
        sys.executable,
        "-m", "demucs.separate",
        "-n", DEMUCS_MODEL,
        "--two-stems", "vocals",
        "-d", DEVICE,
        "-j", JOBS,
        "-o", str(OUT_DIR),
        str(audio)
    ]

    try:
        subprocess.run(cmd, check=True)
        status = "ok"
        err = ""
    except subprocess.CalledProcessError as e:
        status = "fail"
        err = str(e)
        print(f"[FAIL_DEMUCS] {audio.name}")

    results.append({
        "file": audio.name,
        "status": status,
        "error": err
    })

# ========= 写 manifest =========
with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["file", "status", "error"])
    writer.writeheader()
    for r in results:
        writer.writerow(r)

with open(JSONL_PATH, "w", encoding="utf-8") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("\nDONE.")
print("CSV :", CSV_PATH)
print("JSONL:", JSONL_PATH)
