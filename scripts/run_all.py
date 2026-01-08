#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os

STEPS = [
    (1, "MIT RIRs", "scripts/01_download_rirs.py"),
    (2, "AudioSet", "scripts/02_download_audioset.py"),
    (3, "FSD50K", "scripts/03_download_fsd50k.py"),
    (4, "MUSAN", "scripts/04_download_musan.py"),
    (5, "FMA", "scripts/05_download_fma.py"),
    (6, "Precomputed features", "scripts/06_download_features.py"),
    (7, "Prepare config", "scripts/07_prepare_config.py"),
    (8, "Generate synthetic clips", "scripts/08_generate_clips.py"),
    (9, "Augment clips", "scripts/09_augment_clips.py"),
    (10, "Train model", "scripts/10_train_model.py"),
    (11, "Convert to TFLite", "scripts/11_convert_tflite.py"),
]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the full openWakeWord training pipeline"
    )
    parser.add_argument(
        "--skip-steps",
        type=str,
        default="",
        help="Comma-separated list of step numbers to skip (e.g. 1,2,6)",
    )
    return parser.parse_args()

def run_step(step_num, name, script):
    print(f"\n===== Step {step_num}: {name} =====")
    if not os.path.exists(script):
        print(f"ERROR: missing {script}")
        sys.exit(1)

    subprocess.check_call([sys.executable, script])

def main():
    args = parse_args()

    skip_steps = set()
    if args.skip_steps:
        try:
            skip_steps = {int(s.strip()) for s in args.skip_steps.split(",") if s.strip()}
        except ValueError:
            print("ERROR: --skip-steps must be a comma separated list of integers")
            sys.exit(1)
    valid_steps = {step[0] for step in STEPS}
    invalid = skip_steps - valid_steps

    if invalid:
        print(f"ERROR: invalid step numbers in --skip-steps: {sorted(invalid)}")
        print(f"Valid steps are: {sorted(valid_steps)}")
        sys.exit(1)

    if skip_steps:
        print(f"Skipping steps: {sorted(skip_steps)}\n")

    print("\nopenWakeWord full training pipeline\n")
    for step_num, name, script in STEPS:
        if step_num in skip_steps:
            print(f"\n===== Step {step_num}: {name} (SKIPPED) =====")
            continue
        run_step(step_num, name, script)

    print("\nPipeline complete.")
    print("Models available in: data/generated/\n")

if __name__ == "__main__":
    main()
