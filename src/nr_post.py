"""Hậu xử lý màu cho ảnh restored: tăng bão hoà giữ nguyên độ sáng (+ tuỳ chọn tương phản) – nhắm khối NR
(UIQM / UCIQE / CCF, trọng số 0.1 trong composite). LB 23/10: CCF của ta thấp nhất bảng ở cả 2 track
(T2 0.023 vs aimslabs 0.056, T1 0.076 vs ~0.09) → output hồi quy về trung bình, nhạt màu.

Phép biến đổi (giống hệt nhau cho zip Track 1 và graph ONNX Track 2):
    Y  = 0.299 R + 0.587 G + 0.114 B                (độ sáng từng pixel)
    x1 = Y + sat · (x − Y)                           (sat > 1: tăng bão hoà, Y giữ nguyên)
    x2 = m + con · (x1 − m),  m = trung bình Y cả ảnh (con = 1: tắt)
    x3 = x2 + usm · (Y − G_σ * Y)                    (unsharp mask trên luma, cộng như nhau cho 3 kênh → giữ màu;
                                                      G_σ Gaussian tách 2 chiều, kernel chuẩn hoá lại ở biên; usm = 0: tắt)
    out = clip(x3, 0, 1)

25/10: sharpness T2 của ta 0.417 vs UW-IPL 0.959 (PSNR còn cao hơn ta) → usm nhắm khối sharp (0.3·0.1);
dev 05/10 cùng chế độ sharp ~0.4: unsharp α 0.5 = +0.008 composite, PSNR chỉ −0.07.

    # Track 1: áp lên zip đã nộp (seg_masks giữ nguyên) → so LB 2 bài cùng mask = tác động riêng của phép màu
    python nr_post.py zip --src submission.zip --dst submission_sat1.3.zip --sat 1.3
    # Track 2: nối vào cuối graph ONNX (chỉ Slice/Mul/Add/Sub/Clip/GlobalAveragePool – không Conv → FLOPs không đổi)
    python nr_post.py onnx --src track2.onnx --dst track2_sat1.3.onnx --sat 1.3
    # + làm nét luma (2 Conv 1 kênh tách 2 chiều + Mul bù biên, không op mới → FLOPs + ~4·k·H·W, k = 2·ceil(3σ)+1)
    python nr_post.py onnx --src track2.onnx --dst track2_usm0.5.onnx --sat 1.15 --usm 0.5 --usm_sigma 1
    # + bias logit seg từng class (bg, coral, fish, elasmo, other) → đổi argmax; restored không đổi
    python nr_post.py onnx --src track2.onnx --dst track2_b.onnx --sat 1.0 --seg_bias 0,0,0,1,0
    # đo nhanh UCIQE / colorfulness (proxy, CPU) trên N ảnh của zip với nhiều mức sat
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
    """1 / tổng trọng số kernel rơi trong ảnh tại từng vị trí (dài n) – chuẩn hoá lại kernel ở biên."""
    return (1.0 / np.convolve(np.ones(n, np.float32), k, mode="same")).astype(np.float32)


def blur(y, sigma):
    """Gaussian tách 2 chiều, zero-pad rồi chia tổng trọng số trong ảnh (normalized convolution) → biên không tối đi,
    không cần op Pad (khớp Conv pads + Mul vector của ONNX)."""
    k = gauss_kernel(sigma)
    r = len(k) // 2
    h, w = y.shape
    yp = np.pad(y, ((r, r), (0, 0)))
    t = sum(k[i] * yp[i:i + h] for i in range(len(k))) * edge_norm(h, k)[:, None]   # dọc
    tp = np.pad(t, ((0, 0), (r, r)))
    return sum(k[i] * tp[:, i:i + w] for i in range(len(k))) * edge_norm(w, k)[None]  # ngang


def boost(x, sat=1.0, con=1.0, usm=0.0, usm_sigma=1.0):
    """x: float (H, W, 3) trong [0, 1]."""
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
    if step > 1:  # giữ lượng tử hoá bước 2 như shrink_submission (zip < 1024 MB)
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
    assert mb < 1024, "vượt giới hạn upload 1024 MB → tăng --step"
    return mb


# ---------------- Track 2: ONNX ----------------

def post_onnx(src, dst, sat, con=1.0, usm=0.0, usm_sigma=1.0, seg_bias=None):
    """Nối phép màu (+ làm nét luma) vào sau output `restored_image` (đổi tên output cũ thành `restored_raw`)."""
    import onnx
    from onnx import helper, numpy_helper, TensorProto

    m = onnx.load(src)
    g = m.graph
    used = {t.name for t in g.initializer} | {o for n in g.node for o in n.output}
    P = next(q for q in ("cb_", "cb2_", "cb3_", "cb4_") if not any(u.startswith(q) for u in used))  # file đã qua nr_post
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
    for j, k in enumerate("rgb"):  # Slice kênh j → Mul trọng số luma
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
        h, w = (d.dim_value for d in g.input[0].type.tensor_type.shape.dim[2:])  # input tĩnh = kích thước restored
        c(P + "usm", usm)
        c(P + "kv", k.reshape(1, 1, -1, 1))
        c(P + "kh", k.reshape(1, 1, 1, -1))
        c(P + "nv", edge_norm(h, k).reshape(1, 1, h, 1))
        c(P + "nh", edge_norm(w, k).reshape(1, 1, 1, w))
        nodes += [helper.make_node("Conv", [P + "y", P + "kv"], [P + "bv0"], pads=[r, 0, r, 0]),   # dọc
                  helper.make_node("Mul", [P + "bv0", P + "nv"], [P + "bv"]),
                  helper.make_node("Conv", [P + "bv", P + "kh"], [P + "bh0"], pads=[0, r, 0, r]),  # ngang
                  helper.make_node("Mul", [P + "bh0", P + "nh"], [P + "blur"]),
                  helper.make_node("Sub", [P + "y", P + "blur"], [P + "hf"]),
                  helper.make_node("Mul", [P + "hf", P + "usm"], [P + "hfa"]),
                  helper.make_node("Add", [last, P + "hfa"], [P + "x3"])]
        last = P + "x3"
    nodes.append(helper.make_node("Clip", [last, P + "zero", P + "one"], ["restored_image"]))
    bias = np.asarray(seg_bias if seg_bias is not None else [0.0] * 5, dtype=np.float32)
    if bias.any():  # seg_logits + b (Add (1, 5, 1, 1), không FLOPs)
        sp = next(n for n in g.node if "seg_logits" in n.output)
        sp.output[list(sp.output).index("seg_logits")] = P + "seg_raw"
        c(P + "segb", bias.reshape(1, -1, 1, 1))
        nodes.append(helper.make_node("Add", [P + "seg_raw", P + "segb"], ["seg_logits"]))
    if P != "cb_":  # lượt chồng: node không tên → ORT gộp Slice của 2 lượt thành Split trùng tên, lỗi load
        for j, n in enumerate(nodes):     # (lượt đầu giữ không tên để file cũ dựng lại trùng byte)
            n.name = f"{P}n{j}"
    g.node.extend(nodes)
    onnx.checker.check_model(m)
    onnx.save(m, dst)

    try:
        import onnxruntime as ort  # kiểm: output mới = boost(output cũ), seg_logits không đổi
    except ImportError:  # base image của BTC không có onnxruntime → bỏ qua bước tự kiểm (file ghi ra không đổi)
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


# ---------------- proxy NR (chỉ để so tương đối giữa các mức sat) ----------------

def uciqe(x):
    """UCIQE (Yang & Sowmya 2015) trên CIELab: 0.4680 σ_chroma + 0.2745 con_L + 0.2576 μ_sat (Lab chuẩn hoá về [0, 1])."""
    import cv2
    lab = cv2.cvtColor(x.astype(np.float32), cv2.COLOR_RGB2LAB)
    L, a, b = lab[..., 0] / 100.0, lab[..., 1] / 128.0, lab[..., 2] / 128.0
    ch = np.sqrt(a ** 2 + b ** 2)
    lo, hi = np.percentile(L, [1, 99])
    s = ch / np.maximum(np.sqrt(ch ** 2 + L ** 2), 1e-6)
    return 0.4680 * ch.std() + 0.2745 * (hi - lo) + 0.2576 * s.mean()


def colorfulness(x):
    """Hasler & Süsstrunk 2003 (thành phần colorfulness của CCF dùng cùng ý tưởng opponent)."""
    r, g, b = x[..., 0] * 255, x[..., 1] * 255, x[..., 2] * 255
    rg, yb = r - g, 0.5 * (r + g) - b
    return np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)


def probe(src, sats, n=60, con=1.0, seed=0):
    with zipfile.ZipFile(src) as z:
        names = sorted(k for k in z.namelist() if k.startswith("restored/") and not k.endswith("/"))
        pick = np.random.default_rng(seed).choice(len(names), min(n, len(names)), replace=False)
        imgs = [np.asarray(Image.open(io.BytesIO(z.read(names[i]))).convert("RGB"), dtype=np.float32) / 255.0
                for i in sorted(pick)]
    print(f"{len(imgs)} ảnh từ {src}")
    for s in sats:
        out = [boost(x, s, con) for x in imgs]
        u = np.mean([uciqe(o) for o in out])
        cf = np.mean([colorfulness(o) for o in out])
        clip = np.mean([((o <= 0) | (o >= 1)).mean() for o in out])
        dev = np.mean([np.sqrt(((o - x) ** 2).mean()) for o, x in zip(out, imgs)])
        print(f"sat {s:4.2f} con {con:g}: UCIQE {u:.4f}  colorfulness {cf:6.2f}  pixel bị clip {clip:.4f}  "
              f"RMS thay đổi {dev * 255:5.2f}/255 (≈ PSNR tự so {20 * np.log10(1 / max(dev, 1e-9)):.1f} dB)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["zip", "onnx", "probe"])
    p.add_argument("--src", required=True)
    p.add_argument("--dst")
    p.add_argument("--sat", type=float, default=1.3)
    p.add_argument("--con", type=float, default=1.0)
    p.add_argument("--usm", type=float, default=0.0, help="độ mạnh unsharp mask trên luma (0 = tắt)")
    p.add_argument("--usm_sigma", type=float, default=1.0)
    p.add_argument("--seg_bias", default=None, help="bias logit 'b0,b1,b2,b3,b4' (bg, coral, fish, elasmo, other)")
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
