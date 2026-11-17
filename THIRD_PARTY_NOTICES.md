# Third-party notices

This is an unofficial implementation.
Upstream implementations and model parameters retain their original terms.
Please cite the corresponding papers when using these metrics.

## FADE

Lark Kwon Choi, Jaehee You, and Alan C. Bovik.
**Referenceless Prediction of Perceptual Fog Density and Perceptual Image Defogging.**
IEEE Transactions on Image Processing, 24(11), 3888–3901, 2015.

[Paper](https://doi.org/10.1109/TIP.2015.2456502) ·
[Project](https://live.ece.utexas.edu/research/fog/index.html) ·
[Original code](https://live.ece.utexas.edu/research/fog/FADE_release.zip)

The original models and full [copyright notice](dehaze_nr_metrics/fade/data/readme.txt)
are included. The notice also applies to the FADE adaptations and shared
contrast-energy helpers.

## BIQME

Ke Gu, Dacheng Tao, Junfei Qiao, and Weisi Lin.
**Learning a No-Reference Quality Assessment Model of Enhanced Images With Big Data.**
IEEE Transactions on Neural Networks and Learning Systems, 29(4), 1301–1313, 2018.

[Paper](https://arxiv.org/abs/1904.08632) ·
[Author homepage](https://kegu.netlify.app/) ·
[Original code and models](https://kegu.netlify.app/CODE/CODE_2018TNNLS.rar)

The Python implementation follows the published feature extraction and SVR method.
Original parameters and MATLAB P-code are supplied with the
[author's readme](dehaze_nr_metrics/biqme/data/readme.txt).

## Entropy

Entropy uses the Shannon formula on a pooled 256-bin RGB histogram, consistent
with MATLAB's documented [histogram definition](https://www.mathworks.com/help/images/ref/entropy.html).

## Additional credits

- **Phase congruency:** [Peter Kovesi's implementation](https://www.peterkovesi.com/matlabfns/PhaseCongruency/phasecong3.m).
  Its original copyright and license are retained in [pc.py](dehaze_nr_metrics/biqme/pc.py).
- **CDF 9/7 wavelet:** an independent implementation of the lifting equations
  described by [Pascal Getreuer](https://getreuer.info/posts/waveletcdf97/).
- **LIBSVM:** used by the original MATLAB BIQME program; its
  [copyright notice](matlab/BIQME/LIBSVM_COPYRIGHT.txt) is included.
