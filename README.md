# Representation Learning with Quantum Signal Processing

Code and data for the paper *Representation Learning with Quantum Signal
Processing* by Junqi Wang and Junyu Liu (arXiv:2608.28828). Three Python
scripts reproduce the numerical data and Figs. 1 and 2.

| File | Purpose |
| --- | --- |
| [qsp.py](qsp.py) | QSP response and Jacobian, realized kernels, finite-depth formulas, gradient descent, RK4 and the scalar reference. |
| [generate_data.py](generate_data.py) | Experiment settings at the top; generates raw data, statistics and every array the figures need. |
| [plot_figures.py](plot_figures.py) | Styles and captions at the top; reads the saved arrays and writes the figures. |

## Redraw the figures

Use Python 3.10 or newer, from this directory:

```console
python -m pip install -r requirements.txt
python plot_figures.py
```

This reads `data/figure_data.npz` and writes Figs. 1 and 2, with their
captions, to `figures/`. It runs no simulations. The renderings in
`reference_figures/` were produced from the same data; image bytes can differ
slightly across Matplotlib versions.

## Regenerate the data

```console
python generate_data.py
python plot_figures.py
```

The generator starts from fixed random seeds and the model definitions. It does
not read the supplied data. With the default settings it reproduces both
dynamics cells of Fig. 2 and the full occupancy scan, which takes many hours;
the dense cell alone takes several hours. Each run overwrites the files in
`DATA_DIR`, so set a fresh directory (and `DATA_FILE` in the plotting script)
for a separate run.

## Change an experiment or a figure

The opening block of `generate_data.py` holds every experiment setting:
initialization depths and sample counts, the dynamics cells and seeds, the
gradient-descent step factor and step cap, the scan depths, node counts,
horizons and step sizes, and the analysis settings. `GENERATE_INITIALIZATION`,
`GENERATE_DYNAMICS` and `SCAN_STAGES` select which families are computed;
families that are switched off are absent from the new plotting bundle.

The opening block of `plot_figures.py` holds fonts, colors, line and marker
styles, figure sizes, axis limits, output formats and captions. Captions are
written beside the figure files.

## Data

| Path | Contents |
| --- | --- |
| [data/protocol.json](data/protocol.json) | Settings of the production run and its numerical environment. |
| [data/provenance.json](data/provenance.json) | What the supplied data contain and how far they have been reproduced. |
| `data/initialization.npz` | Sampled QSP outputs and diagonal kernels for Fig. 1(a). |
| `data/dynamics/cell_d*_n*.npz` and `.json` | Gradient-descent loss histories, initial conditions, kernel spectra, hitting times and cell settings for Fig. 2(a,b). |
| `data/occupancy_scan/*.jsonl.gz` | One scan point per line, with its parameters and per-seed observables. |
| [data/occupancy_scan/summary.json](data/occupancy_scan/summary.json) | Fits and statistics of the occupancy scan and the Fig. 2(a) hitting-time ratios. |
| [data/occupancy_scan/fig2a_hitting_times.json](data/occupancy_scan/fig2a_hitting_times.json) | Hitting times of the sparse cell in physical time. |
| `data/figure_data.npz` | All 115 arrays read by `plot_figures.py`. |

[NUMERICAL_NOTES.md](NUMERICAL_NOTES.md) defines every observable, time unit,
seed convention and data field.

JSON files open in any text editor. To list every array in the NPZ archives
and inspect the closure-error medians, run in this directory:

```python
from pathlib import Path
import numpy as np

for path in sorted(Path("data").rglob("*.npz")):
    print(path)
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            print(" ", key, data[key].shape, data[key].dtype)

with np.load("data/figure_data.npz", allow_pickle=False) as data:
    print(data["scan/eps/1024/w"])
    print(data["scan/eps/1024/median"])
```

To inspect one scan point and the fit:

```python
import gzip
import json
from pathlib import Path

summary = json.loads(Path("data/occupancy_scan/summary.json").read_text())
print(summary["fit"])
with gzip.open("data/occupancy_scan/dynamic.jsonl.gz", "rt", encoding="utf-8") as stream:
    point = json.loads(next(stream))
print(point.keys())
print(point["d"], point["n"], point["seeds"])
print(point["eps"]["2"])  # One sampled-time maximum per seed at T=2.
```

A fresh generator run writes plain `.jsonl`; read it with `open(...)` instead
of `gzip.open(...)`.

## License and citation

MIT License; see [LICENSE](LICENSE). If you use this code or data, please cite
the paper; see [CITATION.cff](CITATION.cff).
