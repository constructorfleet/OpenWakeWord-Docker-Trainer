import os, numpy as np, datasets, scipy.io.wavfile
import logging
import time
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
    out = os.environ["RIRS_PATH"]
    logger.info("Downloading RIRs into %s", out)
    os.makedirs(out, exist_ok=True)

    ds = datasets.load_dataset(
        "davidscripka/MIT_environmental_impulse_responses",
        split="train",
        streaming=True,
    )

    start = time.perf_counter()
    count = 0
    for row in tqdm(ds):
        name = row["audio"]["path"].split("/")[-1]
        scipy.io.wavfile.write(
            f"{out}/{name}",
            16000,
            (row["audio"]["array"] * 32767).astype("int16"),
        )
        count += 1
    logger.info("Downloaded %d RIR files in %s", count, _fmt_duration(time.perf_counter() - start))


if __name__ == "__main__":
    main()
