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
    config_file = os.environ["CONFIG_FILE"]
    runtime_config_file = os.environ["RUNTIME_CONFIG_FILE"]

    # Prefer runtime config if it already exists so env-derived paths aren't lost.
    config_path = runtime_config_file if os.path.exists(runtime_config_file) else config_file
    logger.info("Loading training config from %s", config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    processed_positive_dirs = []

    positives_path = os.environ["POSITIVES_PATH"]
    logger.info("Processing positives under %s", positives_path)
    preprocess_start = time.perf_counter()
    processed_files = 0
    for p in Path(positives_path).iterdir():
        if not p.is_dir():
            continue
        out_dir = Path(f"{p.absolute()}_16k")
        out_dir.mkdir(parents=True, exist_ok=True)

        for f in p.glob("*.wav"):
            audio, _ = librosa.load(f, sr=16000, mono=True)
            sf.write(out_dir / f.name, audio, 16000)
            processed_files += 1

        processed_positive_dirs.append(str(out_dir))

    config["processed_positive_paths"] = processed_positive_dirs

    with open(runtime_config_file, "w") as f:
        yaml.safe_dump(config, f)
    logger.info("Updated runtime config at %s", runtime_config_file)
    logger.info(
        "Processed %d positive files in %s",
        processed_files,
        _fmt_duration(time.perf_counter() - preprocess_start),
    )

    logger.info("Generating synthetic clips")
    generate_start = time.perf_counter()
    subprocess.check_call([
        sys.executable,
        "openWakeWord/openwakeword/train.py",
        "--training_config", runtime_config_file,
        "--generate_clips",
    ])
    logger.info("Clip generation complete in %s", _fmt_duration(time.perf_counter() - generate_start))


if __name__ == "__main__":
    main()
