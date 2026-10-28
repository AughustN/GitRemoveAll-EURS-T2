# Method description

Team: GitRemoveAll   Track(s): 2 (this repository; Track 1 is in a separate repository)   Reported testing-phase submission(s): #19408 (user #2010)

## Approach

One small joint network, exported as a single ONNX file with a static input of (1, 3, 768, 1024). It has 0.934 M parameters and 9.94 GFLOPs (official script).

- **Encoder:** torchvision MobileNetV3-Large (ImageNet-1k V2), truncated at stride 16. ImageNet normalisation is done inside the graph.
- **Segmentation head:** LR-ASPP. It outputs 5-class `seg_logits` at the input resolution.
- **Restoration head (`mnv3l_r3`):** it predicts a per-cell colour affine transform (3x4) at three scales, H/16, H/8 and H/4. The H/4 stage reads the stride-4 encoder features and the 4x4 average-pooled input colour. Each affine field is upsampled bilinearly and applied to the input, and a full-resolution residual is added for detail. The result is `restored_image`.
- **Post-processing appended as ONNX nodes after the network** (`src/nr_post.py`; it changes no weights):
  1. Luma-preserving saturation `x' = Y + 1.15 (x - Y)`.
  2. Luma unsharp mask `x'' = x' + 1.0 (Y - G_sigma1 * Y)`. It uses a separable Gaussian with sigma = 1, normalised at the borders, built from 2 single-channel Conv nodes. The output is clipped to [0, 1].
  3. A constant per-class bias is added to `seg_logits`: `[0, -1.0, -1.0, +1.0, -0.5]` for (background, coral, fish, elasmobranch, other).

### How the weights were obtained

Every stage was trained on Kaggle (2x T4, PyTorch, AMP).

| Stage | What was trained / done |
|---|---|
| 1. Distillation | The student (`mnv3l`) is trained from ImageNet init for 6000 iterations. Segmentation uses CE + Dice on the labels plus KL distillation (T = 2) from a frozen DINOv3 ViT-L/16 segmentation teacher. Restoration uses the GT losses (PSNR 0.1, cosine 0.5, outlier-aware L1 1.0, SSIM 0.5) plus distillation from a Track 1 RestoreNet teacher (ConvNeXt-T) on restore batches and on segmentation and external images. Augmentations: physically based re-degradation of clean or external images (synthetic water types), geometric and colour augmentation, and minority-class sampling. |
| 2. Fine-tune | Fine-tuned for the official Track 2 harness: bilinear stretch of every image to 768x1024 with no rotation of portrait images. Same losses, lower learning rate. |
| 3. Soup | A weight average of the fine-tuning checkpoints at iterations 500 and 250. |
| 4. KD offline + WiSE-FT | Continued training with teacher soft labels cached on clean images, followed by linear weight interpolation `theta = 0.45 * soup + 0.55 * KD-offline(it 6000)`. |
| 5. Restore r2 | Encoder and segmentation head frozen (BatchNorm in eval mode), so `seg_logits` is unchanged. A wider restoration head (`mnv3l_r2`), zero-initialised, is trained with the GT losses and the restore distillation. |
| 6. Restore r3 (final) | Same as stage 5, plus the zero-initialised H/4 affine stage (`mnv3l_r3`). The final checkpoint is `weights/t2_r3_all_it8000.pt` (iteration 8000). The iteration was chosen on a 16/4 scene validation fold. |

## Training data

- **EURS:** the 20 labelled training scenes. Each stage was run twice:
  - a **fold** run on 16 scenes, validated on `fish13, fish20, fish9, starfish2`, used only to choose iterations and hyper-parameters;
  - an **all** run on all 20 scenes, with the iteration taken from the fold run.

  The submitted weights come from the *all* run. The development-phase and testing-phase images were **not** used for training, pseudo-labelling or adaptation.
- **External data** (stages 1–6, sampled with probability 0.4 per segmentation sample, 30,619 images in total). Labels were remapped to the 5 EURS classes, and ambiguous categories were set to ignore.
  - SUIM (1,598 images, MIT)
  - USIS10K (5,215 images, Apache-2.0)
  - UIIS10K (10,036 images, Apache-2.0)
  - USIS16K (11,695 images, research use)
  - Coralscapes (2,075 images, Apache-2.0)
- **Pretrained models:**
  - torchvision MobileNetV3-Large ImageNet-1k V2 (student initialisation).
  - Teacher models, used only during training and not part of the submission:
    - Segmentation teacher: `facebook/dinov3-vitl16-pretrain-lvd1689m` (frozen) with a segmentation head trained on EURS plus the external data above. It is the same teacher as our Track 1 segmentation model.
    - Restoration teacher: our Track 1 RestoreNet, with a timm ConvNeXt-Tiny ImageNet backbone, trained on EURS.

## Anything non-standard at inference

The graph has no test-time augmentation and no ensemble. It contains the post-processing listed above: saturation 1.15, luma unsharp mask (alpha 1.0, sigma 1), and the constant seg-logit bias.

**Disclosure:** the post-processing constants were chosen by comparing testing-phase leaderboard scores. The network weights were fixed from #19189 on; the later submissions (#19192 to #19408) differed only in these constants, and we report the best one:
- saturation 1.15 (compared with 1.0 and 1.3);
- unsharp strength 1.0 and sigma 1 (compared with 0.5, 1.5 and sigma 0.7);
- the five per-class logit biases, in steps of 0.5.

Each is one fixed value applied identically to every image.

No test image was used to train, fine-tune or adapt any network weight.

## Reproduction notes

- `python repro_check.py . --export-onnx` regenerates the ONNX on the CPU in under one minute. It needs only the base image; no extra packages are required.
- **The regenerated file is not byte-identical to `onnx/submitted_model.onnx`, and the checker reports "differs".** We tested this with PyTorch 2.4.0, 2.7.1 and 2.11.0 (2.11.0 is the version used on Kaggle to produce the submitted file). The result was the same in every case:
  - The graph is identical: 529 nodes with the same names and the same initializer names.
  - Only the Conv weights produced by constant-folding BatchNorm differ, by a few float32 ULP (max abs 5.7e-6). This comes from CPU floating-point arithmetic during folding, not from the checkpoint.
  - Outputs on ONNX Runtime CPU: `seg_logits` max diff 8e-6 with 100% argmax agreement; `restored_image` max diff 2.4e-6, where about 3e-5 of the pixels differ by one 8-bit grey level.
- The checkpoint's SHA-256 is `94912c7d863184a41cdddf75047b8139f6de42f0f6e010c89c5585ec283d958f`. It is a plain `torch.save` dict with the keys `model`, `it`, `hw` and `arch`.
- Leaderboard efficiency for #19408: 34.0 QPS and about 351 mJ/frame on the Jetson AGX Orin.
- Training code (not needed for reproduction): available on request.
