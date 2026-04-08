import torchaudio
from pathlib import Path
import csv
import statistics

# ========= 路径 =========
ROOT = Path(__file__).resolve().parents[1]

IN_DIR = ROOT / "work" / "vad_segments"
OUT_DIR = ROOT / "output_dataset" / "wavs"
MANIFEST_DIR = ROOT / "output_dataset"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_CSV = MANIFEST_DIR / "dataset_manifest.csv"

# ========= RVC 参数 =========
TARGET_SR = 40000          # 和 RVC 训练一致
MIN_DURATION_SEC = 0.1     # 你当前策略：<1s 丢弃

# ========= 统计 =========
durations = []
kept = 0
skipped_short = 0

print(f"[INFO] Exporting from: {IN_DIR}")
print(f"[INFO] Exporting to  : {OUT_DIR}")
print(f"[INFO] Target SR     : {TARGET_SR}")
print(f"[INFO] Min duration  : {MIN_DURATION_SEC}s")

# ========= 打开 CSV =========
with open(MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "filename",
        "source_folder",
        "duration_sec",
        "sample_rate"
    ])

    # ========= 遍历每个子文件夹 =========
    for folder in sorted(p for p in IN_DIR.iterdir() if p.is_dir()):
        prefix = folder.name
        print(f"[FOLDER] {prefix}")

        idx = 1
        wavs = sorted(folder.glob("*.wav"))

        for wav_path in wavs:
            audio, sr = torchaudio.load(wav_path)

            # 单声道
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0, keepdim=True)

            duration = audio.shape[-1] / sr
            if duration < MIN_DURATION_SEC:
                skipped_short += 1
                continue

            # 重采样
            if sr != TARGET_SR:
                audio = torchaudio.functional.resample(audio, sr, TARGET_SR)
                sr = TARGET_SR

            out_name = f"{prefix}_{idx:06d}.wav"
            out_path = OUT_DIR / out_name

            torchaudio.save(out_path, audio, sr)

            # 记录
            durations.append(duration)
            writer.writerow([
                out_name,
                prefix,
                round(duration, 3),
                sr
            ])

            kept += 1
            idx += 1

# ========= 输出统计 =========
print("\n===== EXPORT SUMMARY =====")

if durations:
    print(f"Kept segments        : {kept}")
    print(f"Skipped (<1s)        : {skipped_short}")
    print(f"Total duration (h)   : {sum(durations)/3600:.2f}")
    print(f"Min duration (sec)   : {min(durations):.2f}")
    print(f"Max duration (sec)   : {max(durations):.2f}")
    print(f"Mean duration (sec)  : {statistics.mean(durations):.2f}")
    print(f"Median duration (sec): {statistics.median(durations):.2f}")
else:
    print("No valid segments exported!")

print(f"\nCSV manifest saved to: {MANIFEST_CSV}")
print("DONE.")
