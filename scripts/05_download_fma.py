import os
import logging
from pathlib import Path
from utils import get_dataset_as_16k_audio


logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    os.makedirs(os.environ["FMA16K_PATH"], exist_ok=True)

    logger.info("Streaming FMA (small) and resampling to 16kHz")
    n_hours = 5
    get_dataset_as_16k_audio(
        dataset_name="rudraml/fma",
        display_text="FMA",
        data_dir=Path(os.environ["FMA16K_PATH"]),
        logger=logger,
        split="train",
        ds_range=range(n_hours * 3600 // 30),
        logged_units=f"files (~{n_hours} hours)"
    )


if __name__ == "__main__":
    main()
