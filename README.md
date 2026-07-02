# Gain-Aware Recursive Controller

This repository provides the training, evaluation, mask refinement, and dataset preparation code for a lightweight prediction-space recursive controller for polyp segmentation.

Install:

```bash
pip install -r requirements.txt
```

Data:
- `docs/datasets.md`

Training:

```bash
python main.py --preset paper_main --data-root ./data/light/kvasirseg
python main.py --preset paper_variant_global --data-root ./data/light/kvasirseg
```

Validation:

```bash
python scripts/eval.py --preset paper_main --checkpoint ./runs/checkpoints/paper_main.pth --data-root ./data/light/kvasirseg
```

Mask Refinement:

```bash
python scripts/postprocess.py --input-dir predictions/raw --output-dir predictions/refined
```
