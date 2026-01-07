import sys
import os
import subprocess
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
    logger.info("Augmenting clips")
    start = time.perf_counter()
    subprocess.check_call([
        sys.executable,
        "openWakeWord/openwakeword/train.py",
        "--training_config", os.environ["RUNTIME_CONFIG_FILE"],
        "--augment_clips",
    ])
    logger.info("Augmentation complete in %s", _fmt_duration(time.perf_counter() - start))


if __name__ == "__main__":
    main()
