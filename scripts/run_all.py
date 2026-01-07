#!/usr/bin/env python3
import subprocess
import sys
import os

STEPS = [
    ("MIT RIRs", "scripts/01_download_rirs.py"),
    ("AudioSet", "scripts/02_download_audioset.py"),
    ("FSD50K", "scripts/03_download_fsd50k.py"),
    ("MUSAN", "scripts/04_download_musan.py"),
    ("FMA", "scripts/05_download_fma.py"),
    ("Precomputed features", "scripts/06_download_features.py"),
    ("Prepare config", "scripts/07_prepare_config.py"),
    ("Generate synthetic clips", "scripts/08_generate_clips.py"),
    ("Augment clips", "scripts/09_augment_clips.py"),
    ("Train model", "scripts/10_train_model.py"),
    ("Convert to TFLite", "scripts/11_convert_tflite.py"),
]

def run_step(name, script):
    print(f"\n===== {name} =====")
    if not os.path.exists(script):
        print(f"ERROR: missing {script}")
        sys.exit(1)

    subprocess.check_call([sys.executable, script])

def main():
    print("\nopenWakeWord full training pipeline\n")
    for name, script in STEPS:
        run_step(name, script)

    print("\nPipeline complete.")
    print("Models available in: data/generated/\n")

if __name__ == "__main__":
    main()