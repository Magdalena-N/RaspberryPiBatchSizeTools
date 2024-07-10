import os
import logging
import re
import numpy as np

import tensorflow as tf

MODELS_DIR = "saved_models"
OUTPUT_DIR = "converted_models"

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    filename='convert.log',
                    filemode='w')
console = logging.StreamHandler()
logger = logging.getLogger("convert")
logger.addHandler(console)

def convert(model_dir, save_dir):
    converter = tf.lite.TFLiteConverter.from_saved_model(model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converted_tflite_model = converter.convert()
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    open(save_dir, "wb").write(converted_tflite_model)


if __name__ == "__main__":
    for ver in os.listdir(MODELS_DIR):
        logger.info(f"Currently converting version {ver}")
        for model in os.listdir(os.path.join(MODELS_DIR, ver)):
            if os.path.exists(os.path.join(OUTPUT_DIR, ver, model.replace("_frozen", ".tflite"))):
                logger.info(f"{model} already in converted folder, skipping")
                continue
            logger.info(f"Converting {model} to TFLite")
            try:
                convert(os.path.join(MODELS_DIR, ver, model),
                        os.path.join(OUTPUT_DIR, ver, model.replace("_frozen", "_quant.tflite")))
                logger.info(f"Successfully converted model {model} to TFLite")
            except Exception as e:
                logger.error(f"Couldn't convert model {model} with exception: {str(e)}")

