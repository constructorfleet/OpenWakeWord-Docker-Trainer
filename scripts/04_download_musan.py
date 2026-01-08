import os, tarfile, subprocess
import logging
import time
from pathlib import Path
import datasets, scipy.io.wavfile
from tqdm import tqdm
from scripts.utils import decode_audio_to_16k_int16, infer_filename


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
    os.makedirs(os.environ["MUSAN_PATH"], exist_ok=True)
    os.makedirs(os.environ["MUSAN16K_PATH"], exist_ok=True)

    tar_path = f"{os.environ['MUSAN_PATH']}/musan.tar.gz"
    if not os.path.exists(tar_path):
        logger.info("Downloading MUSAN archive")
        download_start = time.perf_counter()
        subprocess.run([
            "wget", "-O", tar_path,
            "https://openslr.org/resources/17/musan.tar.gz"
        ], check=True)
        logger.info("MUSAN archive downloaded in %s", _fmt_duration(time.perf_counter() - download_start))

    extract_start = time.perf_counter()
    with tarfile.open(tar_path) as tar:
        tar.extractall(os.environ["MUSAN_PATH"])
    logger.info("MUSAN extracted in %s", _fmt_duration(time.perf_counter() - extract_start))
    files = list(Path(os.environ["MUSAN_PATH"]).glob("**/*.wav"))
    logger.info("Resampling %d MUSAN files to 16kHz", len(files))
    ds = datasets.Dataset.from_dict({"audio": [str(f) for f in files]})
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16000))

    resample_start = time.perf_counter()
    processed = 0
    for row in tqdm(ds, desc="MUSAN → 16k"):
        pcm, sr = decode_audio_to_16k_int16(row["audio"])
        name = infer_filename(row)
        scipy.io.wavfile.write(
            f"{os.environ['MUSAN16K_PATH']}/{Path(row).stem}.wav",
            sr,
            pcm,
        )
        processed += 1
    logger.info("Processed %d files in %s", processed, _fmt_duration(time.perf_counter() - resample_start))


if __name__ == "__main__":
    main()
