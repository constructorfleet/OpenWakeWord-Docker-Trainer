import onnx, tempfile, tensorflow as tf
from onnx_tf.backend import prepare
import os, yaml
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
    cfg = yaml.safe_load(open(os.environ["RUNTIME_CONFIG_FILE"]))
    model = cfg["model_name"]

    onnx_path = f"{os.environ['GENERATED_PATH']}/{model}.onnx"
    out = f"{os.environ['GENERATED_PATH']}/{model}.tflite"
    logger.info("Converting %s to TFLite", onnx_path)
    start = time.perf_counter()
    tf_rep = prepare(onnx.load(onnx_path), device="CPU")
    with tempfile.TemporaryDirectory() as d:
        tf_rep.export_graph(d)
        conv = tf.lite.TFLiteConverter.from_saved_model(d)
        with open(out, "wb") as f:
            f.write(conv.convert())
    logger.info("Wrote %s in %s", out, _fmt_duration(time.perf_counter() - start))


if __name__ == "__main__":
    main()
