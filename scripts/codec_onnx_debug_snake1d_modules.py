import torch
import torch.onnx
import torch.nn as nn
from pathlib import Path

# from dac.model.dac import DAC
from lac.model.lac import LAC as DAC


def debug_snake1d_modules(model, prefix=""):
    """Debug Snake1d modules to understand alpha structure"""
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        if module.__class__.__name__ == "Snake1d":
            print(f"\nFound Snake1d at: {full_name}")

            if hasattr(module, "alpha"):
                alpha_attr = getattr(module, "alpha")
                print(f"  Alpha type: {type(alpha_attr)}")

                if isinstance(alpha_attr, torch.Tensor):
                    print(f"  Alpha shape: {alpha_attr.shape}")
                    print(f"  Alpha numel: {alpha_attr.numel()}")
                    # print(f"  Alpha values: {alpha_attr}")
                elif hasattr(alpha_attr, "data"):
                    print(f"  Alpha data shape: {alpha_attr.data.shape}")
                    print(f"  Alpha data: {alpha_attr.data}")
                # else:
                # print(f"  Alpha value: {alpha_attr}")
            else:
                print(f"  No alpha attribute found")
        else:
            debug_snake1d_modules(module, full_name)


# Add this before replacement to debug
print("Debugging Snake1d modules...")
codec_ckpt = (Path(__file__).parent / "../models/vampnet/codec.pth").resolve()
device = "cpu"
lac_model = DAC.load(codec_ckpt)
lac_model.to(device)
lac_model.eval()

debug_snake1d_modules(lac_model)
