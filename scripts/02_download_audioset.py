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
    os.makedirs(os.environ["AUDIOSET_PATH"], exist_ok=True)
    os.makedirs(os.environ["AUDIOSET16K_PATH"], exist_ok=True)


    for split in ["bal_train", "unbal_train"]:
        get_dataset_as_16k_audio(
            dataset_name="agkphysics/AudioSet",
            display_text="AudioSet",
            data_dir=Path(os.environ["AUDIOSET16K_PATH"]),
            subset="full",
            logger=logger,
            file_suffix=f"_{split}",
            split=split,
        )


if __name__ == "__main__":
    main()
