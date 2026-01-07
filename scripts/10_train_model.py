import subprocess
import sys
import os
from pathlib import Path
import librosa
import soundfile as sf
import yaml
import logging
import time


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
    logger.info("Loading runtime config from %s", os.environ["RUNTIME_CONFIG_FILE"])
    config = yaml.safe_load(open(os.environ["RUNTIME_CONFIG_FILE"], "r").read())

    from shutil import copyfile

    generated_positive_dir = Path(config["output_dir"]) / "positive_clips"

    assert generated_positive_dir.exists(), "Positive clip directory not found!"

    if "processed_positive_paths" in config:
        logger.info("Injecting real positive examples into %s", generated_positive_dir)
        inject_start = time.perf_counter()
        injected = 0
        for pos_dir in config["processed_positive_paths"]:
            for f in Path(pos_dir).glob("*.wav"):
                dst = generated_positive_dir / f"real_{f.name}"
                if not dst.exists():
                    copyfile(f, dst)
                    injected += 1

    logger.info("Real positive examples injected into training set")
    if "processed_positive_paths" in config:
        logger.info("Injected %d files in %s", injected, _fmt_duration(time.perf_counter() - inject_start))

    logger.info("Training model")
    train_start = time.perf_counter()
    subprocess.check_call([
        sys.executable,
        "openWakeWord/openwakeword/train.py",
        "--training_config", os.environ["RUNTIME_CONFIG_FILE"],
        "--train_model",
    ])
    logger.info("Training complete in %s", _fmt_duration(time.perf_counter() - train_start))


if __name__ == "__main__":
    main()
