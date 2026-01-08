from asyncio.log import logger
import torch
import numpy as np
import torchaudio
import datasets
import scipy.io.wavfile
from tqdm import tqdm
from pathlib import Path
import time
from typing import Any, Sequence, Tuple
from logging import Logger

TARGET_SR = 16000


def decode_audio_to_16k_int16(audio_obj: Any) -> Tuple[np.ndarray, int]:
    """
    Decode audio from either:
      - torchcodec AudioDecoder
      - legacy HF datasets audio dict

    Returns:
      (pcm_int16_np, 16000)
    """

    # TorchCodec path
    if hasattr(audio_obj, "get_all_samples"):
        samples = audio_obj.get_all_samples()
        wave = samples.data            # Tensor [C, T]
        sr = samples.sample_rate
    else:
        # Legacy datasets backend
        wave = torch.from_numpy(audio_obj["array"])
        sr = int(audio_obj["sampling_rate"])
        if wave.ndim == 1:
            wave = wave.unsqueeze(0)

    wave = wave.to(torch.float32)

    # Mono
    if wave.shape[0] > 1:
        wave = wave.mean(dim=0, keepdim=True)

    # Resample
    if sr != TARGET_SR:
        wave = torchaudio.functional.resample(wave, sr, TARGET_SR)

    wave = wave.squeeze(0).clamp(-1.0, 1.0)
    pcm = (wave * 32767).to(torch.int16).cpu().numpy()

    return pcm, TARGET_SR


def infer_filename(row: dict, default_ext=".wav") -> str:
    """
    Infer a stable filename from a dataset row.
    Falls back to a hash-safe name if needed.
    """

    # Common patterns
    if "audio" in row and isinstance(row["audio"], dict) and "path" in row["audio"]:
        return row["audio"]["path"].split("/")[-1]

    if "path" in row:
        return str(row["path"]).split("/")[-1]

    if "file" in row:
        return str(row["file"]).split("/")[-1]

    if "id" in row:
        return f"{row['id']}{default_ext}"

    # Absolute fallback
    return f"sample_{abs(hash(str(row))) % (10**10)}{default_ext}"


def format_duration(seconds: float) -> str:
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def get_dataset_as_16k_audio(
        dataset_name: str,
        display_text: str,
        data_dir: Path,
        logger: Logger,
        subset: str | None = None,
        split: str | None = None,
        file_prefix: str = "",
        file_suffix: str = "",
        logged_units: str = "files",
        ds_range: Sequence[int] | None = None,
) -> datasets.Dataset:
    """Download and decode a dataset's audio to 16k WAV files."""
    ds = datasets.load_dataset(
        dataset_name,
        subset,
        split=split,
        streaming=True,
    )

    start = time.perf_counter()
    count = 0

    for row in tqdm(ds if ds_range is None else ds_range, desc=f"{display_text} → 16k WAV"):
        row = row if ds_range is None else next(ds)
        decoder = row["audio"]
        pcm, sr = decode_audio_to_16k_int16(decoder)

        name = f"{file_prefix}{Path(infer_filename(row)).stem}{file_suffix}.wav"
        scipy.io.wavfile.write(
            data_dir.joinpath(name),
            sr,
            pcm,
        )
        count += 1
    logger.info(
        "Downloaded %d %s in %s",
        count,
        logged_units,
        format_duration(time.perf_counter() - start)
    )