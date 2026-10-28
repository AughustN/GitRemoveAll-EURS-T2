`submitted_model.onnx` is the exact file uploaded to the leaderboard for testing-phase submission #19408
(SHA-256 `01e5916557bd5bd3e31a0f83fd15c8a55596fce8e62b2d6493ca27fe4039df23`).

`python repro_check.py . --export-onnx` regenerates it from `weights/` with `export_onnx.py`. The result has the same graph and
differs only by float32 rounding in the BatchNorm-folded weights; see "Reproduction notes" in `METHOD.md`.
