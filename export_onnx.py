"""Track 2: regenerate the submitted ONNX file from the trained weights.

The organizers call   main(model_dir, out_onnx).   Every step between the checkpoint and the uploaded file is here:

  1. EURSLite (MobileNetV3-Large encoder + LR-ASPP segmentation head + affine restoration head `_r3`) is rebuilt
     from weights/t2_r3_all_it8000.pt and exported with torch.onnx.export (opset 17, constant folding, static shape).
  2. nr_post pass 1: luma-preserving saturation x' = Y + 1.15 (x - Y) on `restored_image`, then clip to [0, 1].
  3. nr_post pass 2: luma unsharp mask (alpha 1.0, Gaussian sigma 1, border-normalised) on `restored_image`, and a
     constant per-class bias added to `seg_logits`.

Steps 2 and 3 only append ONNX nodes (Slice/Mul/Add/Sub/Conv/Clip) after the network outputs; no weights change.
"""
import inspect
import os
import tempfile

import torch

from src.eurs_lite import EURSLite
from src.nr_post import post_onnx

CKPT = "t2_r3_all_it8000.pt"
SAT = 1.15                                   # pass 1
USM, USM_SIGMA = 1.0, 1.0                    # pass 2
SEG_BIAS = [0.0, -1.0, -1.0, 1.0, -0.5]       # pass 2: background, coral, fish, elasmobranch, other
OPSET = 17


def _export(model, path, hw):
    x = torch.rand(1, 3, *hw)
    kw = dict(input_names=["input"], output_names=["restored_image", "seg_logits"],
              opset_version=OPSET, do_constant_folding=True)
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:  # torch >= 2.5; 2.4 always uses the TorchScript exporter
        kw["dynamo"] = False
    torch.onnx.export(model, (x,), path, **kw)


def main(model_dir, out_onnx):
    torch.manual_seed(0)
    ck = torch.load(os.path.join(model_dir, CKPT), map_location="cpu", weights_only=False)
    hw = tuple(ck.get("hw", (576, 768)))
    model = EURSLite(hw=hw, dynamic=False, arch=ck.get("arch", "lite"), pretrained=False).eval()
    model.load_state_dict(ck["model"])
    os.makedirs(os.path.dirname(os.path.abspath(out_onnx)), exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        raw, sat = os.path.join(tmp, "raw.onnx"), os.path.join(tmp, "sat.onnx")
        _export(model, raw, hw)
        post_onnx(raw, sat, sat=SAT)
        post_onnx(sat, out_onnx, sat=1.0, usm=USM, usm_sigma=USM_SIGMA, seg_bias=SEG_BIAS)
    return out_onnx


if __name__ == "__main__":
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    main(os.path.join(here, "weights"), sys.argv[1] if len(sys.argv) > 1 else "export.onnx")
