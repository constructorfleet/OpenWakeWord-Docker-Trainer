import os, tarfile, subprocess
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
    os.makedirs(os.environ["AUDIOSET_PATH"], exist_ok=True)
    os.makedirs(os.environ["AUDIOSET16K_PATH"], exist_ok=True)

    tars = [f"bal_train{i:02d}.tar" for i in range(10)] + \
           [f"unbal_train{i:02d}.tar" for i in range(20)]

    logger.info("Downloading/extracting AudioSet archives")
    download_start = time.perf_counter()
    for f in tars:
        p = f"{os.environ['AUDIOSET_PATH']}/{f}"
        if not os.path.exists(p):
            subprocess.run([
                "wget", "-O", p,
                f"https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/{f}"
            ], check=True)
        with tarfile.open(p) as t:
            t.extractall(os.environ["AUDIOSET_PATH"])
    logger.info("AudioSet archives ready in %s", _fmt_duration(time.perf_counter() - download_start))

    files = list(Path(f"{os.environ['AUDIOSET_PATH']}/audio").glob("**/*.flac"))
    logger.info("Resampling %d files to 16kHz", len(files))
    resample_start = time.perf_counter()
    ds = datasets.Dataset.from_dict({"audio": [str(f) for f in files]})
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16000))

    processed = 0
    for r in tqdm(ds):
        scipy.io.wavfile.write(
            f"{os.environ['AUDIOSET16K_PATH']}/{Path(r['audio']['path']).stem}.wav",
            16000,
            (r["audio"]["array"] * 32767).astype("int16"),
        )
        processed += 1
    logger.info("Processed %d files in %s", processed, _fmt_duration(time.perf_counter() - resample_start))


if __name__ == "__main__":
    main()
