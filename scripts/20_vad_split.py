import torch
import torchaudio
import soundfile as sf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DENOISE_DIR = ROOT / "work" / "denoise"
OUT_DIR = ROOT / "work" / "vad_segments_40k"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VAD_SR = 16000
OUT_SR = 40000

MIN_SEGMENT_SEC = 1.5
MAX_SEGMENT_SEC = 12.0
PAD_SEC = 0.25

DEVICE = torch.device("cpu")

model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
(get_speech_timestamps, _, _, _, _) = utils
model = model.to(DEVICE).eval()

vocals = list(DENOISE_DIR.rglob("vocals.wav"))
print(f"[INFO] Found {len(vocals)} vocals.wav")

gain = 1.0  # 建议先 1.0，确认音质没问题再加 2.0~2.5

for wav_path in vocals:
    print(f"[VAD] {wav_path}")

    # 1) 读原始音频（保留原采样率）
    audio_hi, sr_hi = torchaudio.load(wav_path)

    # 单声道：建议选左声道，避免 mean 相位抵消
    if audio_hi.shape[0] > 1:
        audio_hi = audio_hi[0:1]
    audio_hi = audio_hi.squeeze(0)  # (T,)

    # 2) 给 VAD 专用的 16k 音频
    if sr_hi != VAD_SR:
        audio_vad = torchaudio.functional.resample(audio_hi.unsqueeze(0), sr_hi, VAD_SR).squeeze(0)
    else:
        audio_vad = audio_hi

    speech_ts = get_speech_timestamps(
        audio_vad.to(DEVICE),
        model,
        sampling_rate=VAD_SR,
        threshold=0.6,
        min_speech_duration_ms=250,
        min_silence_duration_ms=200
    )

    if not speech_ts:
        continue

    out_subdir = OUT_DIR / wav_path.parent.name
    out_subdir.mkdir(parents=True, exist_ok=True)

    for idx, seg in enumerate(speech_ts):
        # 3) 把 16k 的时间戳映射回原 sr 的索引
        start_hi = int(seg["start"] * sr_hi / VAD_SR)
        end_hi   = int(seg["end"]   * sr_hi / VAD_SR)

        # pad 也用原 sr
        start_hi = max(0, start_hi - int(PAD_SEC * sr_hi))
        end_hi   = min(len(audio_hi), end_hi + int(PAD_SEC * sr_hi))

        dur = (end_hi - start_hi) / sr_hi
        if dur < MIN_SEGMENT_SEC or dur > MAX_SEGMENT_SEC:
            continue

        chunk_hi = audio_hi[start_hi:end_hi].unsqueeze(0)  # (1, T) 原 sr

        # 4) 统一输出到 40k
        if sr_hi != OUT_SR:
            chunk_40k = torchaudio.functional.resample(chunk_hi, sr_hi, OUT_SR)
        else:
            chunk_40k = chunk_hi

        # 5) （可选）固定增益 + 防削顶（建议最后再开）
        if gain != 1.0:
            chunk_40k = chunk_40k * gain
            peak = chunk_40k.abs().max().item()
            if peak > 0.98:
                chunk_40k = chunk_40k * (0.98 / (peak + 1e-9))

        out_path = out_subdir / f"{idx:04d}.wav"
        sf.write(
            str(out_path),
            chunk_40k.squeeze(0).cpu().numpy().astype("float32"),
            OUT_SR,
            subtype="FLOAT"   # 32-bit float
        )

print("DONE: VAD split (save=40k) finished.")
