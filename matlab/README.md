# MATLAB comparison

Original FADE `.m` files and BIQME `.p`/LIBSVM files are in `FADE/` and `BIQME/`.
Entropy uses MATLAB's built-in `entropy(I)` on the RGB image. No MathWorks source is redistributed.
Tested with MATLAB **R2023a (9.14.0.2206163)**.

From the project root:

```matlab
addpath('matlab');
fade_scores = test_fade('examples/Fattal/CoA++');
biqme_scores = test_biqme('examples/Fattal/CoA++');
entropy_scores = test_entropy('examples/Fattal/CoA++');
```

Each file prints the mean and returns per-image scores. Replace `CoA++` with
`Input` or use your own folder. Omit the argument to use the included CoA++ images.

```matlab
writetable(fade_scores, 'fade_scores.csv');
```
