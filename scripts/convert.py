#!/usr/bin/env python3
import os
import argparse
import sys
import requests
from urllib.parse import urlparse
from rknn.api import RKNN
import logging
import onnx

# --- Configuration ---
INPUT_DIR = "/workspace/input_models"
OUTPUT_DIR = "/workspace/output_models"

DEFAULT_QUANT_DTYPE = "w8a8"

TARGET_PLATFORMS = {
    "RV1103", "RV1103b", "RV1106", "RV1106b", "RV1126b",
    "RK2118", "RK3562", "RK3566", "RK3568", "RK3576", "RK3588"
}
DEFAULT_TARGET_PLATFORM = "RK3566"

DEFAULT_SHAPES = [(128, 32)]

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# --- Helper Functions ---
def download_model(url: str, target_dir: str) -> str | None:
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        if not filename:
            filename = "downloaded_model.onnx"

        target_path = os.path.join(target_dir, filename)
        os.makedirs(target_dir, exist_ok=True)

        logging.info(f"Downloading model from {url} to {target_path}...")
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logging.info("Download complete.")
        return target_path
    except Exception as e:
        logging.error(f"Failed to download model from {url}: {e}")
        return None


def parse_resolutions(res_string: str) -> list[tuple[int, int]] | None:
    shapes = []
    try:
        pairs = res_string.strip().split(',')
        for pair in pairs:
            if not pair: continue
            w_str, h_str = pair.strip().split('x')
            width = int(w_str)
            height = int(h_str)
            if width <= 0 or height <= 0:
                raise ValueError("Width and Height must be positive integers.")
            shapes.append((width, height))
        if not shapes:
            raise ValueError("No valid resolutions found in the string.")
        return shapes
    except ValueError as e:
        logging.error(f"Invalid format in resolutions string '{res_string}': {e}")
        return None


def parse_args():
    p = argparse.ArgumentParser(description="Convert ONNX Grayscale model to RKNN")
    p.add_argument("--model_source", required=True, help="URL or local filename")
    p.add_argument("--target_platform", default=DEFAULT_TARGET_PLATFORM, choices=TARGET_PLATFORMS)
    p.add_argument("--resolutions", help="WxH format, e.g. 128x32")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args()


def get_onnx_input_name(model_path: str) -> str | None:
    try:
        onnx_model = onnx.load(model_path)
        if not onnx_model.graph.input:
            return None
        return onnx_model.graph.input[0].name
    except Exception as e:
        logging.error(f"Failed to load ONNX model {model_path}: {e}")
        return None


# --- Main Conversion Logic ---
def main():
    args = parse_args()
    onnx_model_path = None
    errors_occurred = False

    target_platform = args.target_platform.lower()

    source = args.model_source
    if source.startswith("http://") or source.startswith("https://"):
        onnx_model_path = download_model(source, INPUT_DIR)
        if not onnx_model_path:
            sys.exit(1)
    else:
        local_path = os.path.join(INPUT_DIR, source)
        if os.path.exists(local_path):
            onnx_model_path = local_path
        else:
            logging.error(f"Local model file not found: {local_path}")
            sys.exit(1)

    onnx_input_name = get_onnx_input_name(onnx_model_path)
    if not onnx_input_name:
        logging.error("Could not determine ONNX input name. Aborting.")
        sys.exit(1)

    target_shapes = []
    if args.resolutions:
        parsed_shapes = parse_resolutions(args.resolutions)
        if parsed_shapes:
            target_shapes = parsed_shapes
        else:
            sys.exit(1)
    else:
        target_shapes = DEFAULT_SHAPES

    base_model_name = os.path.splitext(os.path.basename(onnx_model_path))[0]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    quant_dtype = DEFAULT_QUANT_DTYPE

    for width, height in target_shapes:
        rknn = None
        try:
            shape_str = f"{width}x{height}"
            logging.info(f"--- Converting shape: {shape_str} ---")
            rknn = RKNN(verbose=args.verbose)

            logging.info("[1/4] Configuring RKNN for Grayscale (1 channel)...")
            # ИСПРАВЛЕНИЕ: передаем 1 значение для 1 канала (Grayscale)
            rknn.config(
                target_platform=target_platform,
                quantized_dtype=quant_dtype,
                mean_values=[[0]],
                std_values=[[255]],
                optimization_level=1
            )

            logging.info(f"[2/4] Loading ONNX model with 1 channel [1, 1, {height}, {width}]...")
            # ИСПРАВЛЕНИЕ: подставляем 1 канал вместо 3
            ret = rknn.load_onnx(
                model=onnx_model_path,
                inputs=[onnx_input_name],
                input_size_list=[[1, 1, height, width]] 
            )
            if ret != 0: raise RuntimeError(f"RKNN load_onnx failed with code {ret}")

            logging.info("[3/4] Building RKNN model...")
            ret = rknn.build(do_quantization=False)
            if ret != 0: raise RuntimeError(f"RKNN build failed with code {ret}")

            output_filename = f"{base_model_name}_{args.target_platform}_{shape_str}.rknn"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            logging.info(f"[4/4] Exporting RKNN model to: {output_path}")
            ret = rknn.export_rknn(output_path)
            if ret != 0: raise RuntimeError(f"RKNN export_rknn failed with code {ret}")
            logging.info(f"✅ Successfully exported RKNN model for shape {shape_str}!")

        except Exception as e:
            logging.error(f"❌ FAILED to convert shape {shape_str}: {e}")
            errors_occurred = True
        finally:
            if rknn:
                rknn.release()

    if errors_occurred:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
