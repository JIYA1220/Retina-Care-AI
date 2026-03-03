"""
export_onnx.py
Converts PyTorch .pth weights to ONNX format for 3x faster CPU inference.
"""

import torch
import onnx
import onnxruntime as ort
import os
from model.model import build_model

def export_to_onnx(model_name="efficientnet_b0", 
                   pth_path="model/efficientnet_b0_dr.pth", 
                   onnx_path="model/efficientnet_b0_dr.onnx"):
    
    if not os.path.exists(pth_path):
        print(f"Error: {pth_path} not found. Train the model first.")
        return

    print(f"Loading PyTorch weights from {pth_path}...")
    device = torch.device("cpu")
    model = build_model(model_name=model_name, pretrained=False)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.eval()

    # Dummy input for tracing (Batch size 1, 3 channels, 160x160)
    dummy_input = torch.randn(1, 3, 160, 160)

    print(f"Exporting to ONNX: {onnx_path}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )

    # Verify
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("✔ ONNX model exported and verified successfully.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="efficientnet_b0")
    args = parser.parse_args()
    
    pth = f"model/{args.model}_dr.pth"
    onnx_p = f"model/{args.model}_dr.onnx"
    export_to_onnx(args.model, pth, onnx_p)
