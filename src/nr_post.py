"""Colour post-processing of the restored image: luma-preserving saturation (+ optional contrast) and a luma unsharp
mask. It targets the no-reference part of the score (UIQM / UCIQE / CCF, weight 0.1) and the sharpness term; our
outputs were less saturated and softer than those of other teams.

The transform (the same for a Track 1 zip and for the Track 2 ONNX graph):
    Y  = 0.299 R + 0.587 G + 0.114 B                (per-pixel luma)
    x1 = Y + sat · (x − Y)                           (sat > 1: more saturation, Y unchanged)
    x2 = m + con · (x1 − m),  m = mean Y of the image (con = 1: off)
    x3 = x2 + usm · (Y − G_σ * Y)                    (unsharp mask on luma, added equally to the 3 channels so the
                                                      colour is kept; G_σ is a separable Gaussian renormalised at
                                                      the borders; usm = 0: off)
    out = clip(x3, 0, 1)

    # Track 1: apply to a submitted zip (seg_masks unchanged), so two LB scores with the same masks isolate the colour step
    python nr_post.py zip --src submission.zip --dst submission_sat1.3.zip --sat 1.3
    # Track 2: append to the end of the ONNX graph (Slice/Mul/Add/Sub/Clip/GlobalAveragePool only, no Conv)
    python nr_post.py onnx --src track2.onnx --dst track2_sat1.3.onnx --sat 1.3
    # + luma sharpening (2 separable single-channel Convs + a border Mul; about 4·k·H·W FLOPs, k = 2·ceil(3σ)+1)
    python nr_post.py onnx --src track2.onnx --dst track2_usm0.5.onnx --sat 1.15 --usm 0.5 --usm_sigma 1
    # + per-class seg logit bias (bg, coral, fish, elasmo, other): changes the argmax, restored image unchanged
    python nr_post.py onnx --src track2.onnx --dst track2_b.onnx --sat 1.0 --seg_bias 0,0,0,1,0
    # quick UCIQE / colourfulness proxy (CPU) on N images of a zip for several saturation values
    python nr_post.py probe --src submission.zip --sats 1,1.15,1.3,1.5 --n 60
"""
import argparse
import io
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

LUMA = (0.299, 0.587, 0.114)


def gauss_kernel(sigma):
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    return (k / k.sum()).astype(np.float32)


def edge_norm(n, k):
    """1 / (sum of the kernel weights that fall inside the image) at each of n positions: renormalises the kernel at the borders."""
    return (1.0 / np.convolve(np.ones(n, np.float32), k, mode="same")).astype(np.float32)


def blur(y, sigma):
    """Separable Gaussian: zero-pad, then divide by the weight inside the image (normalised convolution), so the borders
    do not darken and no Pad op is needed (matches ONNX Conv pads + a vector Mul)."""
    k = gauss_kernel(sigma)
    r = len(k) // 2
    h, w = y.shape
    yp = np.pad(y, ((r, r), (0, 0)))
    t = sum(k[i] * yp[i:i + h] for i in range(len(k))) * edge_norm(h, k)[:, None]   # vertical
    tp = np.pad(t, ((0, 0), (r, r)))
    return sum(k[i] * tp[:, i:i + w] for i in range(len(k))) * edge_norm(w, k)[None]  # horizontal


def boost(x, sat=1.0, con=1.0, usm=0.0, usm_sigma=1.0):
    """x: float (H, W, 3) in [0, 1]."""
    y = x @ np.asarray(LUMA, dtype=x.dtype)
    out = y[..., None] + sat * (x - y[..., None])
    if con != 1.0:
        m = y.mean()
        out = m + con * (out - m)
    if usm:
        out = out + usm * (y - blur(y, usm_sigma))[..., None]
    return np.clip(out, 0.0, 1.0)


# ---------------- Track 1: zip ----------------

def _zip_one(args):
    name, data, sat, con, step = args
    if not name.startswith("restored/"):
        return name, data
    a = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.float32) / 255.0
    a = np.round(boost(a, sat, con) * 255.0)
    if step > 1:  # keep the step-2 quantisation of shrink_submission (zip < 1024 MB)
        a = (a + step // 2) // step * step
    b = io.BytesIO()
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(b, "PNG", compress_level=9)
    return name, b.getvalue()


def post_zip(src, dst, sat, con=1.0, step=2, workers=3, batch=32):
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout, \
            ProcessPoolExecutor(workers) as ex:
        names = sorted(n for n in zin.namelist() if not n.endswith("/"))
        for i in range(0, len(names), batch):
            chunk = [(n, zin.read(n), sat, con, step) for n in names[i:i + batch]]
            for n, d in ex.map(_zip_one, chunk):
                zout.writestr(n, d)
            if (i // batch) % 10 == 0:
                print(f"  {i + len(chunk)}/{len(names)}", flush=True)
    mb = os.path.getsize(dst) / 2 ** 20
    print(f"{dst}: {mb:.0f} MB ({len(names)} file), sat {sat:g} con {con:g}", flush=True)
    assert mb < 1024, "over the 1024 MB upload limit: increase --step"
    return mb


# ---------------- Track 2: ONNX ----------------

def post_onnx(src, dst, sat, con=1.0, usm=0.0, usm_sigma=1.0, seg_bias=None):
    """Append the colour transform (+ luma sharpening) after the `restored_image` output (the old output is renamed `restored_raw`)."""
    import onnx
    from onnx import helper, numpy_helper, TensorProto

    m = onnx.load(src)
    g = m.graph
    used = {t.name for t in g.initializer} | {o for n in g.node for o in n.output}
    P = next(q for q in ("cb_", "cb2_", "cb3_", "cb4_") if not any(u.startswith(q) for u in used))  # file already processed by nr_post
    raw = "restored_raw" if P == "cb_" else P + "raw"
    prod = next(n for n in g.node if "restored_image" in n.output)
    prod.output[list(prod.output).index("restored_image")] = raw
    c = lambda name, v: g.initializer.append(numpy_helper.from_array(np.asarray(v, dtype=np.float32), name))
    i64 = lambda name, v: g.initializer.append(numpy_helper.from_array(np.asarray(v, dtype=np.int64), name))
    for k, v in ((P + "sat", sat), (P + "zero", 0.0), (P + "one", 1.0)) + (((P + "con", con),) if con != 1.0 else ()):
        c(k, v)
    for k, w in zip("rgb", LUMA):
        c(P + f"w{k}", w)
    i64(P + "axes", [1])
    nodes = []
    for j, k in enumerate("rgb"):  # Slice channel j -> Mul by its luma weight
        i64(P + f"s{k}", [j])
        i64(P + f"e{k}", [j + 1])
        nodes += [helper.make_node("Slice", [raw, P + f"s{k}", P + f"e{k}", P + "axes"], [P + f"{k}"]),
                  helper.make_node("Mul", [P + f"{k}", P + f"w{k}"], [P + f"{k}w"])]
    nodes += [helper.make_node("Add", [P + "rw", P + "gw"], [P + "rg"]),
              helper.make_node("Add", [P + "rg", P + "bw"], [P + "y"]),            # (1, 1, H, W)
              helper.make_node("Sub", [raw, P + "y"], [P + "d"]),
              helper.make_node("Mul", [P + "d", P + "sat"], [P + "ds"]),
              helper.make_node("Add", [P + "y", P + "ds"], [P + "x1"])]
    last = P + "x1"
    if con != 1.0:
        nodes += [helper.make_node("GlobalAveragePool", [P + "y"], [P + "m"]),    # (1, 1, 1, 1)
                  helper.make_node("Sub", [P + "x1", P + "m"], [P + "dm"]),
                  helper.make_node("Mul", [P + "dm", P + "con"], [P + "dmc"]),
                  helper.make_node("Add", [P + "m", P + "dmc"], [P + "x2"])]
        last = P + "x2"
    if usm:
        k = gauss_kernel(usm_sigma)
        r = len(k) // 2
        h, w = (d.dim_value for d in g.input[0].type.tensor_type.shape.dim[2:])  # static input = size of restored
        c(P + "usm", usm)
        c(P + "kv", k.reshape(1, 1, -1, 1))
        c(P + "kh", k.reshape(1, 1, 1, -1))
        c(P + "nv", edge_norm(h, k).reshape(1, 1, h, 1))
        c(P + "nh", edge_norm(w, k).reshape(1, 1, 1, w))
        nodes += [helper.make_node("Conv", [P + "y", P + "kv"], [P + "bv0"], pads=[r, 0, r, 0]),   # vertical
                  helper.make_node("Mul", [P + "bv0", P + "nv"], [P + "bv"]),
                  helper.make_node("Conv", [P + "bv", P + "kh"], [P + "bh0"], pads=[0, r, 0, r]),  # horizontal
                  helper.make_node("Mul", [P + "bh0", P + "nh"], [P + "blur"]),
                  helper.make_node("Sub", [P + "y", P + "blur"], [P + "hf"]),
                  helper.make_node("Mul", [P + "hf", P + "usm"], [P + "hfa"]),
                  helper.make_node("Add", [last, P + "hfa"], [P + "x3"])]
        last = P + "x3"
    nodes.append(helper.make_node("Clip", [last, P + "zero", P + "one"], ["restored_image"]))
    bias = np.asarray(seg_bias if seg_bias is not None else [0.0] * 5, dtype=np.float32)
    if bias.any():  # seg_logits + b (Add of a (1, 5, 1, 1) constant, no FLOPs)
        sp = next(n for n in g.node if "seg_logits" in n.output)
        sp.output[list(sp.output).index("seg_logits")] = P + "seg_raw"
        c(P + "segb", bias.reshape(1, -1, 1, 1))
        nodes.append(helper.make_node("Add", [P + "seg_raw", P + "segb"], ["seg_logits"]))
    if P != "cb_":  # stacked pass: with unnamed nodes ORT fuses the Slices of both passes into Splits with clashing names and fails to load
        for j, n in enumerate(nodes):     # (the first pass keeps unnamed nodes so older files rebuild byte-identically)
            n.name = f"{P}n{j}"
    g.node.extend(nodes)
    onnx.checker.check_model(m)
    onnx.save(m, dst)

    try:
        import onnxruntime as ort  # self-check: new output = boost(old output), seg_logits + bias
    except ImportError:  # the organizers' base image has no onnxruntime: skip the self-check (the written file is the same)
        return dst
    x = np.random.rand(*[d.dim_value for d in g.input[0].type.tensor_type.shape.dim]).astype(np.float32)
    feed = lambda p: ort.InferenceSession(p, providers=["CPUExecutionProvider"]).run(
        ["restored_image", "seg_logits"], {g.input[0].name: x})
    (r0, s0), (r1, s1) = feed(src), feed(dst)
    ref = boost(r0[0].transpose(1, 2, 0), sat, con, usm, usm_sigma).transpose(2, 0, 1)
    dr, ds = float(np.abs(r1[0] - ref).max()), float(np.abs(s1 - (s0 + bias.reshape(1, -1, 1, 1))).max())
    print(f"{dst}: sat {sat:g} con {con:g} usm {usm:g} sigma {usm_sigma:g} | max diff vs numpy {dr:.2e}, "
          f"seg_logits diff {ds:.1e} (bias {bias.tolist()})", flush=True)
    assert dr < 1e-4 and ds < 1e-5 and (bias.any() or ds == 0.0)
    return dst


# ---------------- NR proxies (only for relative comparison between saturation values) ----------------

def uciqe(x):
    """UCIQE (Yang & Sowmya 2015) on CIELab: 0.4680 σ_chroma + 0.2745 con_L + 0.2576 μ_sat (Lab scaled to [0, 1])."""
    import cv2
    lab = cv2.cvtColor(x.astype(np.float32), cv2.COLOR_RGB2LAB)
    L, a, b = lab[..., 0] / 100.0, lab[..., 1] / 128.0, lab[..., 2] / 128.0
    ch = np.sqrt(a ** 2 + b ** 2)
    lo, hi = np.percentile(L, [1, 99])
    s = ch / np.maximum(np.sqrt(ch ** 2 + L ** 2), 1e-6)
    return 0.4680 * ch.std() + 0.2745 * (hi - lo) + 0.2576 * s.mean()


def colorfulness(x):
    """Hasler & Süsstrunk 2003 (the colourfulness term of CCF uses the same opponent-colour idea)."""
    r, g, b = x[..., 0] * 255, x[..., 1] * 255, x[..., 2] * 255
    rg, yb = r - g, 0.5 * (r + g) - b
    return np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)


def probe(src, sats, n=60, con=1.0, seed=0):
    with zipfile.ZipFile(src) as z:
        names = sorted(k for k in z.namelist() if k.startswith("restored/") and not k.endswith("/"))
        pick = np.random.default_rng(seed).choice(len(names), min(n, len(names)), replace=False)
        imgs = [np.asarray(Image.open(io.BytesIO(z.read(names[i]))).convert("RGB"), dtype=np.float32) / 255.0
                for i in sorted(pick)]
    print(f"{len(imgs)} images from {src}")
    for s in sats:
        out = [boost(x, s, con) for x in imgs]
        u = np.mean([uciqe(o) for o in out])
        cf = np.mean([colorfulness(o) for o in out])
        clip = np.mean([((o <= 0) | (o >= 1)).mean() for o in out])
        dev = np.mean([np.sqrt(((o - x) ** 2).mean()) for o, x in zip(out, imgs)])
        print(f"sat {s:4.2f} con {con:g}: UCIQE {u:.4f}  colorfulness {cf:6.2f}  clipped pixels {clip:.4f}  "
              f"RMS change {dev * 255:5.2f}/255 (≈ PSNR vs input {20 * np.log10(1 / max(dev, 1e-9)):.1f} dB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["zip", "onnx", "probe"])
    p.add_argument("--src", required=True)
    p.add_argument("--dst")
    p.add_argument("--sat", type=float, default=1.3)
    p.add_argument("--con", type=float, default=1.0)
    p.add_argument("--usm", type=float, default=0.0, help="strength of the luma unsharp mask (0 = off)")
    p.add_argument("--usm_sigma", type=float, default=1.0)
    p.add_argument("--seg_bias", default=None, help="logit bias 'b0,b1,b2,b3,b4' (bg, coral, fish, elasmo, other)")
    p.add_argument("--step", type=int, default=2)
    p.add_argument("--sats", default="1,1.15,1.3,1.5")
    p.add_argument("--n", type=int, default=60)
    a = p.parse_args()
    if a.mode == "zip":
        post_zip(a.src, a.dst, a.sat, a.con, a.step)
    elif a.mode == "onnx":
        post_onnx(a.src, a.dst, a.sat, a.con, a.usm, a.usm_sigma,
                  [float(v) for v in a.seg_bias.split(",")] if a.seg_bias else None)
    else:
        probe(a.src, [float(s) for s in a.sats.split(",")], a.n, a.con)
