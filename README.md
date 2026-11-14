
# SudokuScan — COMP482 Computer Vision

SudokuScan extracts a digital 9x9 Sudoku grid from a photograph.

## Requirements

- Python 3.10+
- Tesseract OCR
- OpenCV
- NumPy
- pytesseract

## Installation

```bash
pip install -r requirements.txt
```

Install Tesseract OCR separately and ensure it is available on PATH.

## Run on one image

```bash
python sudoku_scan.py --input photo.jpg --outdir out/
```

## Batch processing

```bash
python sudoku_scan.py --batch images_dir/ --outdir results/
```

## Experiments

```bash
python sudoku_scan.py --batch images_dir/ --outdir experiments/ --analyze --annotations dataset/annotations/ --harris-study
```

## Pipeline

1. Preprocessing and illumination normalization
2. Harris corner detection and automatic corner selection
3. A3, A4 and H4 rectification
4. Affine versus homography comparison
5. Cell extraction and digit recognition
6. Sudoku validity checking

## Outputs

For each image, the program produces:

- `_grid.txt`
- `_result.json`
- `_harris.png`
- `_rect_A3.png`
- `_rect_A4.png`
- `_rect_H4.png`
- `_debug.png`

Analysis mode also writes experimental results.

## Limitations

The scanner assumes one approximately upright Sudoku.
Large rotations, weak grid borders, clutter and low contrast
may cause incorrect corner selection or OCR errors.
