# Method description

Team: GitRemoveAll   Track(s): 2 (this repository; Track 1 is in a separate repository)   Reported testing-phase submission(s): #19408 (user #2010)

## Approach

One small joint network, exported as a single ONNX file with a static (1, 3, 768, 1024) input: 0.934 M parameters and 9.94 GFLOPs
by the official script.

The encoder is a torchvision MobileNetV3-Large (ImageNet-1k V2) cut at stride 16; ImageNet normalisation happens inside the graph.
An LR-ASPP head on top gives the 5-class `seg_logits` at input resolution. The restoration head (`mnv3l_r3`) predicts a 3x4
colour-affine transform per cell at three scales, H/16, H/8 and H/4. The H/4 stage also reads the stride-4 encoder features and the
4x4 average-pooled input colour. Each field is upsampled bilinearly and applied to the input, and a full-resolution residual adds
detail back. That is `restored_image`.

After the network, `src/nr_post.py` appends a few constant ONNX nodes. No weight is changed.
1. Luma-preserving saturation, `x' = Y + 1.15 (x - Y)`.
2. Luma unsharp mask, `x'' = x' + 1.0 (Y - G_sigma1 * Y)`: separable Gaussian with sigma 1, normalised at the borders, built from
   two single-channel Conv nodes. The result is clipped to [0, 1].
3. A constant bias on `seg_logits`: `[0, -1.0, -1.0, +1.0, -0.5]` for background, coral, fish, elasmobranch, other.

### How the weights were obtained

All stages were trained on Kaggle (2x T4, PyTorch, AMP).

| Stage | What was done |
|---|---|
| 1. Distillation | Student `mnv3l` from ImageNet init, 6000 iterations. Segmentation: CE + Dice on the labels plus KL distillation (T = 2) from a frozen DINOv3 ViT-L/16 segmentation teacher. Restoration: the GT losses (PSNR 0.1, cosine 0.5, outlier-aware L1 1.0, SSIM 0.5) plus distillation from our Track 1 RestoreNet (ConvNeXt-T) on restoration batches and on segmentation and external images. Augmentation: physically based re-degradation of clean or external images, geometric and colour jitter, minority-class sampling. |
| 2. Fine-tune | Fine-tuned for the official Track 2 harness, which stretches every image bilinearly to 768x1024 and does not rotate portrait images. Same losses, lower learning rate. |
| 3. Soup | Weight average of the fine-tuning checkpoints at iterations 250 and 500. |
| 4. KD offline + WiSE-FT | More training on teacher soft labels cached on clean images, then `theta = 0.45 * soup + 0.55 * KD-offline(it 6000)`. |
| 5. Restore r2 | Encoder and segmentation head frozen (BatchNorm in eval mode), so `seg_logits` does not change. A wider, zero-initialised restoration head (`mnv3l_r2`) is trained with the GT losses and the restoration distillation. |
| 6. Restore r3 (final) | As stage 5, plus a zero-initialised H/4 affine stage (`mnv3l_r3`). Final checkpoint `weights/t2_r3_all_it8000.pt` (iteration 8000, picked on the 16/4 scene split). |

Because every new head starts at zero, each of stages 5 and 6 starts from exactly the output of the previous model.

## Training data

EURS: the 20 labelled training scenes. Every stage was run twice: once on 16 scenes with validation on `fish13`, `fish20`, `fish9`,
`starfish2`, only to choose iterations and hyper-parameters, and once on all 20 scenes with the chosen settings. The submitted weights
come from the 20-scene runs. Development-phase and testing-phase images were not used for training, pseudo-labelling or adaptation.

External segmentation data, used in all stages (an external image replaces a segmentation sample with probability 0.4), 30,619
images with labels remapped to the 5 EURS classes and undefined categories set to ignore:

| Dataset | Images | Licence |
|---|---|---|
| SUIM | 1,598 | MIT |
| USIS10K | 5,215 | Apache-2.0 |
| UIIS10K | 10,036 | Apache-2.0 |
| USIS16K | 11,695 | research use only |
| Coralscapes | 2,075 | Apache-2.0 |

Pretrained models:
- torchvision MobileNetV3-Large, ImageNet-1k V2: student initialisation.
- Teachers, used only during training and not part of the submission: `facebook/dinov3-vitl16-pretrain-lvd1689m` (frozen) with a
  segmentation head trained on EURS and the data above, which is the same model as our Track 1 segmentation; and our Track 1
  RestoreNet (timm ConvNeXt-Tiny ImageNet backbone, trained on EURS).

## Anything non-standard at inference

No TTA and no ensemble. The graph contains the post-processing above: saturation 1.15, unsharp mask (alpha 1.0, sigma 1) and the
constant seg-logit bias.

Disclosure: these constants were chosen from testing-phase leaderboard scores. The weights stayed fixed from #19189 on, and the
later submissions (#19192 to #19408) changed only the constants. We kept saturation 1.15 (also tried 1.0 and 1.3), unsharp strength
1.0 with sigma 1 (also tried 0.5 and 1.5, and sigma 0.7) and the five class biases (moved in steps of 0.5), and we report the best
submission. Each constant is one fixed
value applied the same way to every image. No test image was used to train, fine-tune or adapt any weight.

## Reproduction notes

- `python repro_check.py . --export-onnx` regenerates the ONNX on the CPU in under a minute with the base image alone.
- The regenerated file is not byte-identical to `onnx/submitted_model.onnx`, so the checker prints "differs". We got the same result
  with PyTorch 2.4.0, 2.7.1 and 2.11.0 (2.11.0 produced the submitted file on Kaggle):
  - the graph is the same: 529 nodes, same node and initializer names;
  - only the Conv weights produced by constant-folding BatchNorm differ, by a few float32 ULP (max abs 5.7e-6). This comes from CPU
    floating-point arithmetic during folding, not from the checkpoint;
  - on ONNX Runtime CPU, `seg_logits` differ by at most 8e-6 with 100% argmax agreement, and `restored_image` by at most 2.4e-6,
    which moves about 3e-5 of the pixels by one 8-bit level.
- Checkpoint SHA-256 `94912c7d863184a41cdddf75047b8139f6de42f0f6e010c89c5585ec283d958f`; a plain `torch.save` dict with keys
  `model`, `it`, `hw`, `arch`.
- Leaderboard efficiency of #19408 on the Jetson AGX Orin: 34.0 QPS, about 351 mJ per frame.
- Training code is not needed for reproduction; it is available on request.
