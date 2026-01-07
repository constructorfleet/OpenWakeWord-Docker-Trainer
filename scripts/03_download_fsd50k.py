import os
import logging
import time
from pathlib import Path
import datasets, scipy.io.wavfile
from tqdm import tqdm


logger = logging.getLogger(__name__)

def _fmt_duration(seconds: float) -> str:
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def main():
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    os.makedirs(os.environ["FSD50K16K_PATH"], exist_ok=True)

    logger.info("Downloading FSD50K")
    download_start = time.perf_counter()
    ds = datasets.load_dataset("fsd50k", "full", split="train")
    logger.info("FSD50K loaded in %s", _fmt_duration(time.perf_counter() - download_start))
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16000))

    logger.info("Resampling FSD50K to 16kHz")
    resample_start = time.perf_counter()
    processed = 0
    for row in tqdm(ds, desc="FSD50K → 16k"):
        scipy.io.wavfile.write(
            f"{os.environ['FSD50K16K_PATH']}/{Path(row['audio']['path']).stem}.wav",
            16000,
            (row["audio"]["array"] * 32767).astype("int16"),
        )
        processed += 1
    logger.info("Processed %d files in %s", processed, _fmt_duration(time.perf_counter() - resample_start))


if __name__ == "__main__":
    main()
