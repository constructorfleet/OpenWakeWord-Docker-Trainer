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
    out = os.environ["RIRS_PATH"]
    logger.info("Downloading RIRs into %s", out)
    os.makedirs(out, exist_ok=True)
    
    get_dataset_as_16k_audio(
        dataset_name="davidscripka/MIT_environmental_impulse_responses",
        display_text="MIT RIRs",
        data_dir=Path(os.environ["RIRS_PATH"]),
        logger=logger,
        split="train",
    )


if __name__ == "__main__":
    main()
