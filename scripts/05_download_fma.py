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
    os.makedirs(os.environ["FMA16K_PATH"], exist_ok=True)

    logger.info("Streaming FMA (small) and resampling to 16kHz")
    start = time.perf_counter()
    ds = datasets.load_dataset(
        "rudraml/fma",
        name="small",
        split="train",
        streaming=True,
    )

    ds = iter(ds.cast_column("audio", datasets.Audio(sampling_rate=16000)))

    n_hours = 5
    processed = 0
    for _ in tqdm(range(n_hours * 3600 // 30), desc="FMA → 16k"):
        row = next(ds)
        scipy.io.wavfile.write(
            f"{os.environ['FMA16K_PATH']}/{Path(row['audio']['path']).stem}.wav",
            16000,
            (row["audio"]["array"] * 32767).astype("int16"),
        )
        processed += 1
    logger.info(
        "Processed %d files (~%d hours) in %s",
        processed,
        n_hours,
        _fmt_duration(time.perf_counter() - start),
    )


if __name__ == "__main__":
    main()
