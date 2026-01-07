# openWakeWord Custom Model Training (CUDA, Docker)

This project provides a fully self-contained, CUDA-enabled Docker environment for training custom openWakeWord models using the automated training pipeline.

It reproduces the full functionality of the original training notebook:
	•	dataset downloads
	•	audio preprocessing
	•	synthetic clip generation via Piper
	•	augmentation
	•	training
	•	ONNX + TFLite export

…without relying on Colab, conda, or notebook state.

⸻

## Requirements

Host system
	•	Linux (required by Piper TTS)
	•	Docker ≥ 24
	•	docker-compose
	•	NVIDIA GPU
	•	NVIDIA Container Toolkit installed

Verify GPU access:

docker run --rm --gpus all nvidia/cuda:12.6.2-base-ubuntu22.04 nvidia-smi


⸻

## Project structure
```
openwakeword-docker/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
│
├── openWakeWord/              # cloned automatically in image
│
├── configs/
│   ├── model.yaml          # base training config
│   └── model.runtime.yaml  # generated at runtime
│
├── data/
│   ├── rirs/
│   ├── audioset/
│   ├── audioset_16k/
│   ├── fsd50k_16k/
│   ├── musan/
│   ├── musan_16k/
│   ├── fma_16k/
│   ├── features/
│   └── generated/
│
├── scripts/
│   ├── 01_download_rirs.py
│   ├── 02_download_audioset.py
│   ├── 03_download_fsd50k.py
│   ├── 04_download_musan.py
│   ├── 05_download_fma.py
│   ├── 06_download_features.py
│   ├── 07_prepare_config.py
│   ├── 08_generate_clips.py
│   ├── 09_augment_clips.py
│   ├── 10_train_model.py
│   └── 11_convert_tflite.py
```

All generated data lives under data/ and should be mounted as a volume so it persists across container runs.

⸻

## Build the image
```shell
docker compose build
```

⸻

## Enter the container
```shell
docker compose run --rm trainer
```
You will land in /workspace inside the container.

⸻

## End-to-end training workflow

Each step corresponds directly to the original notebook logic.

1. Download room impulse responses
```shell
python scripts/01_download_rirs.py
```
2. Download and resample AudioSet
```shell
python scripts/02_download_audioset.py
```
3. Download FSD50K
```shell
python scripts/03_download_fsd50k.py
```
4. Download and resample MUSAN
```shell
python scripts/04_download_musan.py
```
5. Download and resample FMA
```shell
python scripts/05_download_fma.py
```
6. Download precomputed openWakeWord features
```shell
python scripts/06_download_features.py
```

⸻

### Prepare training configuration

Modify `configs/model.yaml` as desired, then generate the runtime config:

```shell
python scripts/07_prepare_config.py
``` 

The runtime config file is what the training pipeline actually uses. Paths are injected from ENVs, so keep the base config focused on tunables.

Example base config (paths omitted because they are overwritten at runtime):
```yaml
model_name: hey eddie
model_type: dnn
layer_size: 64
target_phrase:
  - hey eddie
positive_paths:
  - positives_teagan
custom_negative_phrases:
  - "hey"
  - "hey ed"
  - "hey andy"
  - "hey daddy"
  - "hey ready"
  - "hey heavy"
  - "heady"
  - "eddy"
feature_data_files:
  ACAV100M_sample: openwakeword_features_ACAV100M_2000_hrs_16bit.npy
false_positive_validation_data_path: validation_set_features.npy
n_samples: 200000
n_samples_val: 20000
batch_n_per_class:
  ACAV100M_sample: 768
  adversarial_negative: 200
  positive: 80
augmentation_batch_size: 32
augmentation_rounds: 2
max_negative_weight: 4000
steps: 1000000
target_accuracy: 0.95
target_false_positives_per_hour: 0.01
target_recall: 0.8
tts_batch_size: 50
```

Runtime-injected paths (via environment variables):
	•	`CONFIG_FILE` (default `/workspace/configs/model.yaml`)
	•	`RUNTIME_CONFIG_FILE` (default `/workspace/configs/model.runtime.yaml`)
	•	`OUTPUT_DIR` (default `/workspace/out`)
	•	`FMA16K_PATH` (default `/workspace/data/fma_16k`)
	•	`AUDIOSET16K_PATH` (default `/workspace/data/audioset_16k`)
	•	`MUSAN16K_PATH` (default `/workspace/data/musan_16k`)
	•	`FSD50K16K_PATH` (default `/workspace/data/fsd50k_16k)
	•	`RIRS_PATH` (default `/workspace/data/rirs`)
	•	`FEATURES_PATH` (default `/workspace/data/features`)
	•	`POSITIVES_PATH` (default `/workspace/data/positives`)

⸻

### Generate synthetic training clips
```shell
python scripts/08_generate_clips.py
```

Uses Piper TTS to synthesize:
	•	positive examples
	•	adversarial phrases
	•	validation samples

This step may take several minutes depending on configuration.

⸻

### Augment clips
```shell
python scripts/09_augment_clips.py
```

Applies:
	•	room impulse responses
	•	background noise
	•	music and speech overlays
	•	gain and timing variation

⸻

### Train the model
```
python scripts/10_train_model.py
```

Features:
	•	early stopping
	•	adaptive batch sampling
	•	cyclical loss weighting
	•	checkpoint averaging

Outputs are written to:  

`data/generated/`  


⸻

### Export to TFLite
```shell
python scripts/11_convert_tflite.py
```

Produces:
	•	<model_name>.onnx
	•	<model_name>.tflite

Both are compatible with openWakeWord inference pipelines.

⸻

#### Notes on CUDA and torchcodec
	•	PyTorch is installed from the official CUDA 12.6 wheels
	•	torchcodec is installed from the same index to guarantee ABI compatibility
	•	ffmpeg is provided by the system image and matches expected SONAMEs
	•	LD_LIBRARY_PATH is explicitly configured to avoid dynamic loader failures

If torchcodec fails to load, something is wrong with your GPU runtime or driver. It is not a configuration issue in this project.

⸻

#### Optional: Jupyter notebook

If you insist on notebooks:
```shell
pip install jupyterlab
jupyter lab --ip=0.0.0.0 --no-browser --allow-root
```
Then open:  
  
`http://localhost:8888`

The scripts can be imported and executed from notebooks without modification.

⸻

## Troubleshooting

### No GPU visible

Ensure:
	•	nvidia-smi works on the host
	•	nvidia-container-toolkit is installed
	•	Docker is restarted after toolkit installation

### Dataset downloads are slow

This is expected. AudioSet and FMA are large. Mount data/ as a volume to avoid re-downloading.

⸻

## Why this exists

Because:
	•	notebooks lie
	•	conda lies louder
	•	reproducibility matters
	•	and debugging dynamic linker errors at 3am is not character building

This setup is boring.
That’s the point.
