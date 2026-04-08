import os
import shutil
from pathlib import Path
import csv

import torch
import torchaudio
import torch.nn.functional as F
from transformers import WavLMModel, Wav2Vec2FeatureExtractor

# ===================== 路径 =====================
ROOT = Path(__file__).resolve().parents[1]
EE_DIR = ROOT / "ee"     # 参考“干净”音频
SUS_DIR = ROOT / "sus"   # 需要筛的音频

OUT_KEEP = ROOT / "work" / "cosine_keep"
OUT_DROP = ROOT / "work" / "cosine_drop"
MANIFEST_DIR = ROOT / "work" / "manifests"
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
OUT_KEEP.mkdir(parents=True, exist_ok=True)
OUT_DROP.mkdir(parents=True, exist_ok=True)

CSV_PATH = MANIFEST_DIR / "cosine_manifest.csv"

# ===================== 参数 =====================
SAMPLE_RATE = 16000

# 相似度阈值：你可以先用 0.80~0.88 试（越高越严）
THRESH = 0.7

# 片段太短容易不稳（尤其“嗯/啊”），建议跳过或直接丢弃
MIN_SEC = 0.7

# 是否保留目录结构（建议 True，方便定位来源）
KEEP_SUBDIR = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===================== 工具函数 =====================
def list_audio_files(folder: Path):
    exts = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}
    return [p for p in folder.rglob("*") if p.suffix.lower() in exts]

def load_mono_16k(path: Path):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    wav = wav.squeeze(0)  # (T,)
    return wav

@torch.no_grad()
def embed_wavlm(model, extractor, wav_1d):
    """
    wav_1d: 1D float tensor on CPU
    returns: 1D embedding on CPU
    """
    # transformers feature extractor expects numpy-like lists; but accepts torch tensor too in many versions.
    inputs = extractor(
        wav_1d,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    out = model(**inputs).last_hidden_state  # (B, T, C)
    emb = out.mean(dim=1).squeeze(0)         # (C,)
    emb = F.normalize(emb, dim=0)
    return emb.detach().cpu()

def ensure_parent(dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)

def rel_to_root_keep_subdir(p: Path, base: Path):
    if not KEEP_SUBDIR:
        return Path(p.name)
    try:
        return p.relative_to(base)
    except Exception:
        return Path(p.name)

# ===================== 主流程 =====================
print("[INFO] Loading WavLM...")

extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(DEVICE)
model.eval()

print("[INFO] Device:", DEVICE)

# 1) 做参考库 embedding：把 ee 里所有音频都算出来，然后求平均向量
ee_files = list_audio_files(EE_DIR)
if not ee_files:
    raise RuntimeError(f"No reference audio found in: {EE_DIR}")

print(f"[INFO] Reference files: {len(ee_files)}")

ref_embs = []
for p in ee_files:
    wav = load_mono_16k(p)
    dur = wav.numel() / SAMPLE_RATE
    if dur < MIN_SEC:
        print(f"[SKIP_REF_TOO_SHORT] {p} ({dur:.2f}s)")
        continue
    emb = embed_wavlm(model, extractor, wav)
    ref_embs.append(emb)

if not ref_embs:
    raise RuntimeError("All reference files are too short. Put longer clean samples into ee/")

ref = torch.stack(ref_embs, dim=0).mean(dim=0)
ref = F.normalize(ref, dim=0)
print(f"[INFO] Reference embedding built from {len(ref_embs)} files")

# 2) 对 sus 逐个算 embedding + cosine
sus_files = list_audio_files(SUS_DIR)
if not sus_files:
    raise RuntimeError(f"No suspect audio found in: {SUS_DIR}")

print(f"[INFO] Suspect files: {len(sus_files)}")
print(f"[INFO] THRESH={THRESH} MIN_SEC={MIN_SEC}")

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["path", "duration_sec", "cosine", "decision", "dst_path"])

    kept = 0
    dropped = 0
    skipped_short = 0

    for p in sus_files:
        wav = load_mono_16k(p)
        dur = wav.numel() / SAMPLE_RATE

        # 太短片段：建议直接丢到 drop 或者单独保存
        if dur < MIN_SEC:
            skipped_short += 1
            rel = rel_to_root_keep_subdir(p, SUS_DIR)
            dst = OUT_DROP / rel
            ensure_parent(dst)
            shutil.copy2(p, dst)
            writer.writerow([str(p), f"{dur:.3f}", "", "DROP_TOO_SHORT", str(dst)])
            continue

        emb = embed_wavlm(model, extractor, wav)
        cos = float(torch.dot(ref, emb).item())

        if cos >= THRESH:
            kept += 1
            rel = rel_to_root_keep_subdir(p, SUS_DIR)
            dst = OUT_KEEP / rel
            ensure_parent(dst)
            shutil.copy2(p, dst)
            writer.writerow([str(p), f"{dur:.3f}", f"{cos:.5f}", "KEEP", str(dst)])
        else:
            dropped += 1
            rel = rel_to_root_keep_subdir(p, SUS_DIR)
            dst = OUT_DROP / rel
            ensure_parent(dst)
            shutil.copy2(p, dst)
            writer.writerow([str(p), f"{dur:.3f}", f"{cos:.5f}", "DROP", str(dst)])

print("[DONE]")
print("KEEP:", kept)
print("DROP:", dropped)
print("DROP_TOO_SHORT:", skipped_short)
print("CSV:", CSV_PATH)
