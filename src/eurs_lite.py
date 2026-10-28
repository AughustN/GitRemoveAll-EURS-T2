"""EURS-Lite: joint restoration + segmentation model for Track 2 (<= 1.0M params, <= 12 GFLOPs).

Input : (1, 3, H, W) RGB float in [0, 1], static H x W (the submitted model uses 768 x 1024; the official harness
        stretches every image to that size and does not rotate portrait images).
Output: restored_image (1, 3, H, W) in [0, 1]; seg_logits (1, 5, H, W) raw logits
        (background, coral, fish, elasmobranch, other).

Only ops that export cleanly to ONNX / TensorRT are used: Conv, BatchNorm (folded into Conv at export), ReLU / Clip,
Sigmoid, Resize (static size), Concat, Add, Mul, AvgPool, GlobalAveragePool, ReduceMean / ReduceMax.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 5


def up(x, size):
    """Bilinear resize to a static (integer) size, which avoids Shape/Gather in the ONNX graph."""
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, k=1, s=1, groups=1, act=True):
        layers = [nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False), nn.BatchNorm2d(cout)]
        if act:
            layers.append(nn.ReLU6(inplace=True))
        super().__init__(*layers)


class DWSep(nn.Sequential):
    """Depthwise 3×3 + pointwise 1×1 (MobileNetV1)."""

    def __init__(self, cin, cout, s=1):
        super().__init__(ConvBNAct(cin, cin, 3, s, groups=cin), ConvBNAct(cin, cout, 1))


class InvertedResidual(nn.Module):
    """MobileNetV2: 1×1 expand -> depthwise 3×3 -> 1×1 project (linear bottleneck), residual when shapes match."""

    def __init__(self, cin, cout, s, t):
        super().__init__()
        hid = cin * t
        layers = [] if t == 1 else [ConvBNAct(cin, hid, 1)]
        layers += [ConvBNAct(hid, hid, 3, s, groups=hid), ConvBNAct(hid, cout, 1, act=False)]
        self.block = nn.Sequential(*layers)
        self.res = s == 1 and cin == cout

    def forward(self, x):
        y = self.block(x)
        return x + y if self.res else y


# (cout, stride, expand, repeat) per stage; the stem is separate
ENC_CFG = [
    [(16, 1, 1, 1)],                                  # H/2
    [(24, 2, 4, 1), (24, 1, 4, 1)],                   # H/4
    [(40, 2, 4, 1), (40, 1, 4, 2)],                   # H/8
    [(80, 2, 4, 1), (80, 1, 4, 2), (96, 1, 4, 1)],    # H/16
    [(128, 2, 4, 1), (128, 1, 4, 2)],                 # H/32
]


class Encoder(nn.Module):
    def __init__(self, cfg=ENC_CFG, stem=16):
        super().__init__()
        self.stem = ConvBNAct(3, stem, 3, 2)
        cin, self.stages, self.channels = stem, nn.ModuleList(), []
        for stage in cfg:
            blocks = []
            for cout, s, t, n in stage:
                for i in range(n):
                    blocks.append(InvertedResidual(cin, cout, s if i == 0 else 1, t))
                    cin = cout
            self.stages.append(nn.Sequential(*blocks))
            self.channels.append(cin)

    def forward(self, x):
        x = self.stem(x)
        feats = []
        for st in self.stages:
            x = st(x)
            feats.append(x)
        return feats  # [H/2, H/4, H/8, H/16, H/32]


class FPPM(nn.Module):
    """Fast pyramid pooling (RSOS-Net / DSOS-Net): chained 5×5 avg-pools + a global pool, wide context almost for free."""

    def __init__(self, cin, cout):
        super().__init__()
        c = cin // 2
        self.cv1 = ConvBNAct(cin, c, 1)
        self.pool = nn.AvgPool2d(5, 1, 2, count_include_pad=False)
        self.cv2 = ConvBNAct(5 * c, cout, 1)

    def forward(self, x, size):
        x1 = self.cv1(x)
        y1 = self.pool(x1)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        g = up(F.adaptive_avg_pool2d(x1, 1), size)
        return self.cv2(torch.cat([x1, y1, y2, y3, g], 1))


class FFM(nn.Module):
    """Feature fusion (RSOS-Net): DW-sep fusion of several levels + channel attention (SE) + spatial attention (CBAM)."""

    def __init__(self, cin, cout, r=4):
        super().__init__()
        self.conv0 = DWSep(cin, cout)
        self.mlp = nn.Sequential(nn.Conv2d(cout, cout // r, 1), nn.ReLU(inplace=True), nn.Conv2d(cout // r, cout, 1))
        self.sp = nn.Conv2d(2, 1, 3, padding=1)

    def forward(self, x):
        x0 = self.conv0(x)
        x = x0 * torch.sigmoid(self.mlp(F.adaptive_avg_pool2d(x0, 1)))
        s = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        return x * torch.sigmoid(self.sp(s)) + x0


class SegHead(nn.Module):
    """Mini-RSOS: LFPN (H/4–H/16) + FPPM (H/32) -> FFM at H/4 -> refinement with the H/2 features -> logits."""

    def __init__(self, ch, d=64, c_ffm=96, c_out=48, n_cls=NUM_CLASSES):
        super().__init__()
        self.lat = nn.ModuleList([ConvBNAct(c, d, 1) for c in ch[1:4]])
        self.fppm = FPPM(ch[4], d)
        self.fpn = nn.ModuleList([DWSep(d, d) for _ in range(3)])
        self.ffm = FFM(4 * d, c_ffm)
        self.refine = DWSep(c_ffm + ch[0], c_out)
        self.cls = nn.Conv2d(c_out, n_cls, 1)
        # training only (dropped at export): deep supervision at H/16 and an edge head at H/4
        self.aux_seg = nn.Conv2d(d, n_cls, 1)
        self.aux_edge = nn.Conv2d(c_ffm, 1, 1)

    def forward(self, f, sizes, train_aux=False):
        # sizes = [H/2, H/4, H/8, H/16, H/32]
        lat = [l(x) for l, x in zip(self.lat, f[1:4])] + [self.fppm(f[4], sizes[4])]
        for i in range(3, 0, -1):  # top-down
            lat[i - 1] = lat[i - 1] + up(lat[i], sizes[i])
        outs = [conv(x) for conv, x in zip(self.fpn, lat[:3])] + [lat[3]]
        outs = [outs[0]] + [up(o, sizes[1]) for o in outs[1:]]
        x = self.ffm(torch.cat(outs, 1))
        y = self.refine(torch.cat([up(x, sizes[0]), f[0]], 1))
        logits = self.cls(y)
        if train_aux:
            return logits, self.aux_seg(lat[2]), self.aux_edge(x)
        return logits


class RestoreHead(nn.Module):
    """Spatially varying colour-affine field (as in HDRNet) + a detail residual.

    Physics: I = J·t + B·(1 − t) => J = I/t − B(1 − t)/t, so J is an affine transform of I whose coefficients depend
    on t and B, which vary slowly in space. The head predicts a 3×3 matrix + bias per cell at H/16 (local) plus a
    whole-image part (global), upsamples it to H × W and applies it to the input; a small residual adds detail back.
    """

    def __init__(self, ch, c_mid=32, c_det=16):
        super().__init__()
        self.local = nn.Sequential(ConvBNAct(ch[3], c_mid, 1), DWSep(c_mid, c_mid), nn.Conv2d(c_mid, 12, 1))
        self.glob = nn.Sequential(nn.Conv2d(ch[4], c_mid, 1), nn.ReLU(inplace=True), nn.Conv2d(c_mid, 12, 1))
        for m in (self.local[-1], self.glob[-1]):  # initialised to the identity transform
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.det_lat = ConvBNAct(ch[1], c_det, 1)
        self.det_mid = DWSep(c_det + ch[0], c_det)
        self.det_out = nn.Sequential(ConvBNAct(c_det + 3, 8, 3), nn.Conv2d(8, 3, 3, padding=1))
        nn.init.zeros_(self.det_out[-1].weight)
        nn.init.zeros_(self.det_out[-1].bias)
        self.register_buffer("eye", torch.tensor([1., 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]).view(1, 12, 1, 1))

    def affine(self, f, sizes, hw, x=None):
        p = self.local(f[3]) + self.glob(F.adaptive_avg_pool2d(f[4], 1)) + self.eye  # (1, 12, H/16, W/16)
        return up(p, hw)

    def forward(self, x, f, sizes, hw):
        p = self.affine(f, sizes, hw, x)
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        color = torch.cat([p[:, 3 * c:3 * c + 1] * r + p[:, 3 * c + 1:3 * c + 2] * g + p[:, 3 * c + 2:3 * c + 3] * b
                           for c in range(3)], 1) + p[:, 9:12]
        d = self.det_mid(torch.cat([up(self.det_lat(f[1]), sizes[0]), f[0]], 1))
        d = self.det_out(torch.cat([up(d, hw), x], 1))
        return torch.clamp(color + d, 0.0, 1.0)


class RestoreHead2(RestoreHead):
    """RestoreHead + extra affine branches for restoration (arch suffix "_r2", using the spare params / FLOPs):
    ctx at H/16 (H/16 features + pooled H/8 features, 2 inverted residuals with 64 channels) and mid at H/8 (H/8
    features + ctx). The output Convs of both branches start at zero, so at iteration 0 the model is exactly the
    previous RestoreHead. The encoder and seg head can stay frozen (restore-only training) because the new
    branches only read features."""

    def __init__(self, ch, c_mid=32, c_det=16, c_ctx=64):
        super().__init__(ch, c_mid, c_det)
        self.ctx = nn.Sequential(ConvBNAct(ch[3] + ch[2], c_ctx, 1), InvertedResidual(c_ctx, c_ctx, 1, 4),
                                 InvertedResidual(c_ctx, c_ctx, 1, 4))
        self.ctx_out = nn.Conv2d(c_ctx, 12, 1)
        self.mid = nn.Sequential(ConvBNAct(ch[2] + c_ctx, c_mid, 1), DWSep(c_mid, c_mid), nn.Conv2d(c_mid, 12, 1))
        for m in (self.ctx_out, self.mid[-1]):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def _ctx_mid(self, f, sizes):
        c = self.ctx(torch.cat([f[3], F.avg_pool2d(f[2], 2)], 1))
        p = self.local(f[3]) + self.glob(F.adaptive_avg_pool2d(f[4], 1)) + self.eye + self.ctx_out(c)
        h = self.mid[:-1](torch.cat([f[2], up(c, sizes[2])], 1))
        return p, h, self.mid[-1](h)

    def affine(self, f, sizes, hw, x=None):
        p, _, m = self._ctx_mid(f, sizes)
        return up(p, hw) + up(m, hw)  # two separate resizes (not one after another) so that iteration 0 equals RestoreHead exactly


class RestoreHead3(RestoreHead2):
    """RestoreHead2 + a finer affine stage at H/4 (arch suffix "_r3"). It reads the H/4 features, the hidden state of
    the mid branch (H/8) and the local input colour (4×4 avg pool), so the affine coefficients can change at H/4
    instead of being interpolated from H/8 (t and B change quickly at object / background edges, where the distance
    changes). The output Conv starts at zero, so iteration 0 equals RestoreHead2 exactly."""

    def __init__(self, ch, c_mid=32, c_det=16, c_ctx=64, c_fine=48):
        super().__init__(ch, c_mid, c_det, c_ctx)
        self.fine = nn.Sequential(ConvBNAct(ch[1] + c_mid + 3, c_fine, 1), InvertedResidual(c_fine, c_fine, 1, 2),
                                  DWSep(c_fine, c_fine), nn.Conv2d(c_fine, 12, 1))
        nn.init.zeros_(self.fine[-1].weight)
        nn.init.zeros_(self.fine[-1].bias)

    def affine(self, f, sizes, hw, x=None):
        p, h, m = self._ctx_mid(f, sizes)
        q = self.fine(torch.cat([f[1], up(h, sizes[1]), F.avg_pool2d(x, 4)], 1))
        return up(p, hw) + up(m, hw) + up(q, hw)


RESTORE_HEADS = {"": RestoreHead, "r2": RestoreHead2, "r3": RestoreHead3}


class HSwish(nn.Module):
    """x·relu6(x+3)/6 written with Add/Clip/Mul/Div (ops known to the official FLOPs script)."""

    def forward(self, x):
        return x * torch.clamp(x + 3.0, 0.0, 6.0) / 6.0


class HSigmoid(nn.Module):
    def forward(self, x):
        return torch.clamp(x + 3.0, 0.0, 6.0) / 6.0


def _swap_hard_act(module):
    """Replace the nn.Hardswish / nn.Hardsigmoid of torchvision by Clip versions (no parameters, so the weights are unchanged)."""
    for name, m in list(module.named_modules()):
        if isinstance(m, (nn.Hardswish, nn.Hardsigmoid)):
            parent = module.get_submodule(name.rsplit(".", 1)[0]) if "." in name else module
            setattr(parent, name.rsplit(".", 1)[-1], HSwish() if isinstance(m, nn.Hardswish) else HSigmoid())


# torchvision MobileNetV3: (number of feature layers kept, layer indices tapped at H/2, H/4, H/8, H/16, "H/32")
# large: cut after layer 12 (112 channels, H/16) -> 0.79M parameters; drops the three 160-channel blocks (1.4M) + conv 960.
#        There is no H/32, so the last tap repeats H/16 (torchvision's own LR-ASPP also runs large at H/16, dilated).
# small: drops the final 1×1 conv 96->576 -> 0.87M parameters, all 5 levels down to H/32.
MNV3_CFG = {"mnv3l": ("large", 13, (1, 3, 6, 12, 12)), "mnv3s": ("small", 12, (0, 1, 3, 8, 11))}


class MNV3Encoder(nn.Module):
    """MobileNetV3 encoder, ImageNet-1k pretrained (torchvision); ImageNet normalisation is done inside (Sub/Div)."""

    def __init__(self, arch="mnv3l", pretrained=True):
        super().__init__()
        import torchvision.models as tvm
        kind, n, self.taps = MNV3_CFG[arch]
        if kind == "large":
            m = tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None)
        else:
            m = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = m.features[:n]
        _swap_hard_act(self.features)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        with torch.no_grad():
            self.channels = [f.shape[1] for f in self.forward(torch.zeros(1, 3, 64, 64))]

    def forward(self, x):
        x = (x - self.mean) / self.std
        out = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            out[i] = x
        return [out[i] for i in self.taps]


class LRASPP(nn.Module):
    """Lite R-ASPP (MobileNetV3, Howard et al. ICCV 2019): the high branch (H/16 or H/32) is a 1×1 conv times a global
    attention, upsampled to H/8 and added to the logits of the low branch (H/8). aux_seg (training only) = the
    high-branch logits on their own."""

    def __init__(self, ch, inter=128, n_cls=NUM_CLASSES, refine=0):
        super().__init__()
        self.cbr = ConvBNAct(ch[4], inter, 1)
        self.scale = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(ch[4], inter, 1, bias=False), nn.Sigmoid())
        self.low_cls = nn.Conv2d(ch[2], n_cls, 1)
        self.high_cls = nn.Conv2d(inter, n_cls, 1)
        self.refine = refine
        if refine:  # refine the logits at H/4 with H/4 features (object edges); the last conv starts at zero, so it starts as LR-ASPP
            self.ref_lat = ConvBNAct(ch[1], refine // 2, 1)
            self.ref = nn.Sequential(DWSep(refine // 2 + n_cls, refine), DWSep(refine, refine), nn.Conv2d(refine, n_cls, 1))
            nn.init.zeros_(self.ref[-1].weight)
            nn.init.zeros_(self.ref[-1].bias)

    def forward(self, f, sizes, train_aux=False):
        hi = self.cbr(f[4]) * self.scale(f[4])
        hi = self.high_cls(up(hi, sizes[2]))
        logits = self.low_cls(f[2]) + hi
        if self.refine:
            logits = up(logits, sizes[1])
            logits = logits + self.ref(torch.cat([logits, self.ref_lat(f[1])], 1))
        if train_aux:
            return logits, hi, None
        return logits

    def full(self, f, sizes, hw, train_aux=False):
        """Logits at input resolution (same as before: up(forward(...), hw))."""
        if train_aux:
            logits, hi, edge = self.forward(f, sizes, True)
            return up(logits, hw), hi, edge
        return up(self.forward(f, sizes), hw)


class LRASPP2(LRASPP):
    """LR-ASPP + an extra decoder (arch flag "s2", using the spare params / FLOPs; the plain seg head has only 30K
    parameters, 0.11 GF): a light FPN H/16 -> H/8 -> H/4 (H/16 + H/8 + H/4 features and the LR-ASPP logits) adds a
    correction to the logits at H/4 (edges, small objects). Zero-initialised output Conv + separate resizes per
    branch, so iteration 0 equals LR-ASPP exactly (seg-only fine-tuning with encoder and restoration frozen).
    Not used in the submitted model."""

    def __init__(self, ch, inter=128, n_cls=NUM_CLASSES, c=64):
        super().__init__(ch, inter, n_cls)
        self.lat16 = ConvBNAct(ch[4], c, 1)
        self.lat8 = ConvBNAct(ch[2], c, 1)
        self.fuse8 = nn.Sequential(DWSep(c, c), DWSep(c, c))
        self.lat4 = ConvBNAct(ch[1], c // 2, 1)
        self.fuse4 = nn.Sequential(DWSep(c + c // 2 + n_cls, c), DWSep(c, c))
        self.out = nn.Conv2d(c, n_cls, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def full(self, f, sizes, hw, train_aux=False):
        logits, hi, _ = LRASPP.forward(self, f, sizes, True)                       # H/8
        d = self.fuse8(self.lat8(f[2]) + up(self.lat16(f[4]), sizes[2]))
        d = self.fuse4(torch.cat([up(d, sizes[1]), self.lat4(f[1]), up(logits, sizes[1])], 1))
        out = up(logits, hw) + up(self.out(d), hw)  # two separate resizes -> iteration 0 = LR-ASPP
        return (out, hi, None) if train_aux else out


class EURSLite(nn.Module):
    """hw: static size used at export (fixed integers in the graph). dynamic=True takes the size from the input at
    run time (training / PyTorch inference at several sizes); export needs dynamic=False.

    arch: "lite" = our own MobileNetV2-style encoder trained from scratch + mini-RSOS SegHead (early version);
          "mnv3l" / "mnv3s" = ImageNet MobileNetV3 large / small + LR-ASPP, distilled from the DINOv3 teacher;
          "mnv3lr" = mnv3l + a logit refinement branch at H/4 (plain LR-ASPP outputs logits at H/8).
          Suffix "_r2" (e.g. "mnv3l_r2"): RestoreHead2 (extra affine branches for restoration).
          Suffix "_r3" (e.g. "mnv3l_r3"): RestoreHead3 (= r2 + a finer affine stage at H/4). The submitted model is "mnv3l_r3".
          Suffix "_s2" (e.g. "mnv3l_r2_s2"): LRASPP2 (extra seg decoder with logits at H/4).
    """

    def __init__(self, hw=(576, 768), enc_cfg=ENC_CFG, seg_d=64, seg_ffm=96, seg_out=48, dynamic=False, arch="lite",
                 pretrained=True):
        super().__init__()
        assert hw[0] % 32 == 0 and hw[1] % 32 == 0, "H and W must be multiples of 32"
        self.hw, self.dynamic, self.arch = tuple(hw), dynamic, arch
        base, *flags = arch.split("_")
        assert set(flags) <= {"r2", "r3", "s2"}, f"unknown arch flags: {flags}"
        if base == "lite":
            self.encoder = Encoder(enc_cfg)
            ch = self.encoder.channels
            self.seg = SegHead(ch, seg_d, seg_ffm, seg_out)
        else:  # "mnv3lr" = mnv3l + logit refinement at H/4 (uses spare FLOPs under 12 G)
            refine = 64 if base.endswith("r") else 0
            self.encoder = MNV3Encoder(base[:-1] if refine else base, pretrained)
            ch = self.encoder.channels
            assert not ("s2" in flags and refine), "s2 cannot be combined with mnv3lr"
            self.seg = LRASPP2(ch) if "s2" in flags else LRASPP(ch, refine=refine)
        self.restore = RESTORE_HEADS[next((k for k in ("r3", "r2") if k in flags), "")](ch)

    def aux_parameters(self):
        """Parameters used only in training (not in the ONNX graph)."""
        return [p for n, p in self.seg.named_parameters() if n.startswith(("aux_seg.", "aux_edge."))]

    def _sizes(self, x):
        H, W = (int(x.shape[-2]), int(x.shape[-1])) if self.dynamic else self.hw
        return (H, W), [(H // s, W // s) for s in (2, 4, 8, 16, 32)]

    def forward(self, x, train_aux=False, heads=("restore", "seg")):
        hw, sizes = self._sizes(x)
        f = self.encoder(x)
        restored = self.restore(x, f, sizes, hw) if "restore" in heads else None
        if "seg" not in heads:
            return restored, None
        if hasattr(self.seg, "full"):  # LR-ASPP / LRASPP2 (mnv3*)
            out = self.seg.full(f, sizes, hw, train_aux)
            return (restored, *out) if train_aux else (restored, out)
        if train_aux:
            logits, aux_seg, aux_edge = self.seg(f, sizes, True)
            return restored, up(logits, hw), aux_seg, aux_edge
        return restored, up(self.seg(f, sizes), hw)


def export_onnx(model, path, hw=(576, 768), opset=17):
    model.eval()
    x = torch.rand(1, 3, *hw)
    torch.onnx.export(model, (x,), path, input_names=["input"], output_names=["restored_image", "seg_logits"],
                      opset_version=opset, do_constant_folding=True, dynamo=False)
    return path


if __name__ == "__main__":
    import sys
    hw = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (576, 768)
    arch = sys.argv[3] if len(sys.argv) > 3 else "lite"
    m = EURSLite(hw, arch=arch, pretrained=False).eval()
    n = lambda mod: sum(p.numel() for p in mod.parameters())
    aux = sum(p.numel() for p in m.aux_parameters())
    print(f"[{arch}] params: total {n(m) / 1e6:.3f}M (export {(n(m) - aux) / 1e6:.3f}M) | encoder {n(m.encoder) / 1e6:.3f}M "
          f"{m.encoder.channels} | seg {(n(m.seg) - aux) / 1e6:.3f}M | restore {n(m.restore) / 1e6:.3f}M")
    with torch.no_grad():
        r, s = m(torch.rand(1, 3, *hw))
    print("outputs:", tuple(r.shape), tuple(s.shape))
