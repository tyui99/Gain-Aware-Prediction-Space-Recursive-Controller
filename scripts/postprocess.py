from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.dataloader import postprocess_feasible_mask


def collect_masks(folder: Path) -> list[Path]:
    exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
    files: list[Path] = []
    for ext in exts:
        files.extend(sorted(folder.glob(ext)))
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser(description='Apply feasibility postprocess to binary prediction masks.')
    parser.add_argument('--input-dir', required=True, help='Directory containing raw masks.')
    parser.add_argument('--output-dir', required=True, help='Directory to save postprocessed masks.')
    parser.add_argument('--threshold', type=int, default=127, help='Binarization threshold for grayscale masks.')
    parser.add_argument('--min-area', type=int, default=200, help='Minimum connected-component area.')
    parser.add_argument('--border-strip', type=int, default=4, help='Border strip width used when removing border components.')
    parser.add_argument('--keep-global-largest-component', action='store_true', help='Keep only the largest connected component globally.')
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_files = collect_masks(input_dir)
    if not mask_files:
        raise RuntimeError(f'No mask files found under {input_dir}')

    for mask_path in mask_files:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f'Failed to read mask: {mask_path}')
        binary = (mask > int(args.threshold)).astype(np.uint8)
        processed = postprocess_feasible_mask(
            binary,
            min_area=int(args.min_area),
            remove_border=True,
            border_strip=int(args.border_strip),
            keep_global_largest_component=bool(args.keep_global_largest_component),
        )
        out = (processed > 0).astype(np.uint8) * 255
        out_path = output_dir / f'{mask_path.stem}.png'
        cv2.imwrite(str(out_path), out)


if __name__ == '__main__':
    main()
