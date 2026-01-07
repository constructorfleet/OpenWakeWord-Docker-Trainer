import os
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
    logger.info("Preparing runtime config from %s", os.environ["CONFIG_FILE"])
    start = time.perf_counter()
    with open(os.environ["CONFIG_FILE"]) as f:
        cfg = yaml.safe_load(f)

    cfg["background_paths"] = [
        os.environ["FMA16K_PATH"],
        os.environ["AUDIOSET16K_PATH"],
        os.environ["MUSAN16K_PATH"],
        os.environ["FSD50K16K_PATH"],
    ]
    cfg["rir_paths"] = [
        os.environ["RIRS_PATH"],
    ]
    cfg["output_dir"] = os.environ["OUTPUT_DIR"]
    cfg["piper_sample_generator_path"] = os.environ["PIPER_SAMPLE_GENERATOR_PATH"]

    with open(os.environ["RUNTIME_CONFIG_FILE"], "w") as f:
        yaml.safe_dump(cfg, f)
    logger.info(
        "Wrote runtime config to %s in %s",
        os.environ["RUNTIME_CONFIG_FILE"],
        _fmt_duration(time.perf_counter() - start),
    )


if __name__ == "__main__":
    main()
