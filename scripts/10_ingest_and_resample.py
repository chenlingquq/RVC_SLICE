import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INP = ROOT / "input_raw"
OUT = ROOT / "work" / "ingested"
OUT.mkdir(parents=True, exist_ok=True)

def ffmpeg_convert(src: Path, dst: Path, sr=40000):
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-ac", "1",
        "-ar", str(sr),
        "-vn",
        str(dst)
    ]
    subprocess.run(cmd, check=True)

exts = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}
for p in INP.glob("*"):
    if p.suffix.lower() not in exts:
        continue
    out_wav = OUT / (p.stem + ".wav")
    print("[INGEST]", p.name, "->", out_wav.name)
    ffmpeg_convert(p, out_wav)

print("Done. Output:", OUT)
