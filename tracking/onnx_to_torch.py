import onnx
from onnx import version_converter
import torch
from onnx2torch import convert
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='Run a train scripts in train_settings.')
    parser.add_argument('--onnx-model', type=str, required=True, help='Name of the train script.')
    parser.add_argument('--save-dir', type=str, help='the directory to save checkpoints and logs')
    # Load the ONNX model.
    args = parser.parse_args()
    converted_model = onnx.load(args.onnx_model)
    # Convert the model to the target version.
    # target_version = 13
    # converted_model = version_converter.convert_version(model, target_version)
    # Convert to torch.
    torch_model = convert(converted_model)
    print(torch_model)
    torch.save(torch_model, os.path.join(args.save_dir, "model.pt"))

if __name__ == '__main__':
    main()