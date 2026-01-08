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
    os.makedirs(os.environ["FSD50K16K_PATH"], exist_ok=True)

    logger.info("Downloading FSD50K")

    get_dataset_as_16k_audio(
        dataset_name="Fhrozen/FSD50k",
        display_text="FSD50K",
        data_dir=Path(os.environ["FSD50K16K_PATH"]),
        logger=logger,
        split="validation",
    )


if __name__ == "__main__":
    main()
