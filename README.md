<div align="center">

<h1>Dehaze-NR-Metrics</h1>

</div>

## :desktop_computer: Environment

- Python 3.12.
- MATLAB reference: **R2023a (9.14.0.2206163)**.

```bash
conda create -n dehaze-nr-metrics python=3.12 -y
conda activate dehaze-nr-metrics
pip install -r requirements.txt
```

## :book: Project Structure

```text
dehaze_nr_metrics/
├── fade/
├── biqme/
├── entropy/
└── common/
examples/Fattal/     # images for test
├── Input/
└── CoA++/
matlab/              # Original implementation
├── FADE/
├── BIQME/
├── test_fade.m
├── test_biqme.m
└── test_entropy.m
test.py              # evaluation
```

## :taxi: Testing

Evaluate either example folder or replace it with your own:

```bash
python test.py examples/Fattal/Input -o results_input
python test.py examples/Fattal/CoA++ -o results_coapp
```

Saves per-image scores and means as CSV/JSON. Use a new output directory each time.

For a single image:

```python
from dehaze_nr_metrics import fade, biqme, entropy

image = "examples/Fattal/CoA++/aerial.png"
print(fade(image), biqme(image), entropy(image))
```

## :computer: MATLAB Testing

Run the three files independently from the project root. Each returns per-image scores and prints the mean.

```matlab
addpath('matlab');
fade_scores = test_fade('examples/Fattal/CoA++');
biqme_scores = test_biqme('examples/Fattal/CoA++');
entropy_scores = test_entropy('examples/Fattal/CoA++');
```

## :pencil2: Results

Mean scores over the same **31 CoA++ outputs on Fattal**, at their original resolution:

| Metric | Python | MATLAB R2023a |
| --- | ---: | ---: |
| FADE ↓ | 0.27459292 | 0.27459292 |
| BIQME ↑ | 0.61713236 | 0.61713239 |
| Entropy ↑ | 7.55640453 | 7.55640453 |

## :clipboard: Acknowledgments

Thanks to the authors of [FADE](https://live.ece.utexas.edu/research/fog/index.html)
and [BIQME](https://kegu.netlify.app/CODE/CODE_2018TNNLS.rar),
and CoA++ outputs.

For more no-reference image quality metrics, see [pyiqa](https://github.com/chaofengc/IQA-PyTorch).  
For learning-based haze density prediction, see [HazDesNet](https://github.com/JiaheZhang/HazDesNet).

See [source notices](THIRD_PARTY_NOTICES.md).
