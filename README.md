# Figure code and data

This directory reproduces the two figures in **Representation Learning with Quantum Signal Processing**.

## Figure map

- **Figure 1 — initialization geometry.** `make_all_figures.py` generates the diagonal-kernel distribution, the exact finite-depth coefficient of variation, and the angular mean-kernel correlation. The Monte Carlo panel uses 4,000 samples with seed `20260812`; the other panels are deterministic.
- **Figure 2 — training dynamics.** The same script reads the two archived cells in `data/dynamics/`, reconstructs the matched initializations, and plots the QSP trajectories, interquartile bands, matched scalar flows, frozen-kernel references, and continuous-flow reference curve.

The manuscript figure PDFs are included in `reference_figures/` for comparison.

## License and citation

The released software and accompanying files are available under the MIT License. Citation metadata is provided in `CITATION.cff`.

## Run

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python make_all_figures.py
```

The figures are written to `generated/` as PDF and PNG files.
`requirements-lock.txt` records the exact clean environment used for release verification.

## Package boundary

This is a redraw package: Figure 1 is generated directly from the released formulas and fixed Monte Carlo seed, while Figure 2 is redrawn from the archived numerical trajectories. It does not rerun the multi-hour QSP training campaign that produced those archives.

## Files

- `make_all_figures.py`: parameters, panel construction, and output commands.
- `qsp_figure_tools.py`: minimal QSP, QNTK, and scalar-flow routines used by the figures.
- `requirements.txt` and `requirements-lock.txt`: supported dependency ranges and the verified environment.
- `LICENSE` and `CITATION.cff`: reuse terms and citation metadata.
- `data/dynamics/`: archived data and protocol metadata for the two Figure 2 cells.
- `DATA_SCHEMA.md`: fields consumed by the plotting script and provenance notes.
- `reference_figures/`: PDFs used in the manuscript.
