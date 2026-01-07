import os, subprocess
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
    os.makedirs(os.environ["FEATURES_PATH"], exist_ok=True)

    urls = [
        "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
        "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy",
    ]

    logger.info("Downloading precomputed features into %s", os.environ["FEATURES_PATH"])
    start = time.perf_counter()
    for u in urls:
        subprocess.run(["wget", "-P", os.environ["FEATURES_PATH"], u], check=True)
    logger.info("Feature download complete in %s", _fmt_duration(time.perf_counter() - start))


if __name__ == "__main__":
    main()
