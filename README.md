# Transformer m-Height Predictor

Course project for CSCE 636 (Deep Learning), Texas A&M, Fall 2025. TensorFlow/Keras, with the exact solver in SciPy.

## Problem

A systematic linear code is described by a matrix `G = [I_k | P]`, where `G` has k rows and n columns and `P` has k rows and n-k columns. The identity part is fixed, so `P` is the only thing that varies.

The m-height of `G` is computed like this: for each codeword, sort its entries by absolute value and divide the largest by the m-th largest. The m-height is the largest such ratio across all codewords. It is always at least 1.

Computing it exactly requires solving one linear program for every combination of two coordinates, a subset of the remaining coordinates, and a sign pattern. The number of linear programs grows quickly with n and m, so it is slow. `compute_hm` in the notebook is that exact solver, and it also produced the training labels.

The task is to predict the m-height from `(n, k, m, P)` with a neural network instead of running the solver.

## Approach

One model per `(k, n, m)` combination, 21 in total, for k in {4, 5, 6}, n in {9, 10}, and m from 2 to n-k. The input size depends on `(k, n)` and the range of outputs depends on m, so separate models were easier than one shared model.

Three details:

- The models predict `log(m-height)` rather than the raw value. Raw m-heights span several orders of magnitude and are skewed. The output layer is `Dense(1, relu)`, so the prediction cannot drop below 0, which corresponds to an m-height of 1, the minimum. The inference function applies `exp` to the output.
- Permuting the columns of `G` does not change the m-height, since the value depends only on sorted absolute values. Each training matrix is turned into four samples: the original plus three random column permutations of `P`.
- Inputs are divided by 100. Entries of `P` are drawn from Uniform(-100, 100), while the positional encoding added to them is in [-1, 1]. Without rescaling, the entries dominate and the positional signal is lost. The same division is applied at training and inference time.

### Model

```
Input (k, n-k)
  -> Reshape to (k*(n-k), 1)    # one token per entry of P
  -> PositionalEmbedding        # fixed sinusoidal encoding, added
  -> TransformerEncoder x 8     # attention + feed-forward, residual and layer norm
  -> GlobalMaxPooling1D
  -> Dropout(0.5) -> Dense(128) -> Dropout(0.5)
  -> Dense(1, relu)             # log of the m-height
```

`PositionalEmbedding` and `TransformerEncoder` are custom layers defined in the notebook, registered as serializable so the saved models reload without extra setup.

2 attention heads, embedding size 64, feed-forward size 32, Adam, MSE loss, batch size 1024, 30 epochs, 30% validation split. 44,489 parameters for k=4, n=9.

Training data was a pooled set of about 10 million labelled matrices shared across the class, split 70/30 for training and validation. Testing used a separate 21,000 matrices generated locally, 1,000 per configuration.

## Results

Scores from the course evaluator on the held-out test set, lower is better. Bold marks m = n-k.

| n, k | m=2 | m=3 | m=4 | m=5 | m=6 |
|---|---|---|---|---|---|
| 9, 4 | 0.174 | 0.197 | 0.633 | **3.138** | |
| 10, 4 | 0.842 | 0.084 | 0.304 | 0.823 | **3.267** |
| 9, 5 | 0.228 | 0.589 | **3.502** | | |
| 10, 5 | 0.103 | 0.323 | 0.972 | **2.974** | |
| 9, 6 | 0.537 | **2.748** | | | |
| 10, 6 | 0.135 | 0.632 | **3.453** | | |

Average: 1.222.

The error is concentrated in one place. Every configuration with m = n-k scores between 2.7 and 3.5, and every other configuration is below 1.0. At the largest m the m-height distribution has a long tail of rare large values; the models fit the common range and miss the tail. Most of the average error comes from those six configurations, so that is the part to work on next.

## Data format

Inputs and outputs are Python lists saved with `pickle.dump()`.

- Input: a list of samples, each `[n, k, m, P]`, where `P` is a numpy array with k rows and n-k columns.
- Output: a list of m-heights, in the same order as the input list.

The provided training set has 32,087 samples with n=9. Test sets use the same format, so the predictions file is a list of numbers matching the input list one for one.

## Usage

```bash
pip install gdown tamu_csce_636_project1 tensorflow scipy joblib pandas matplotlib
```

Run [the notebook](Project_DL636_Mahbub_Alam.ipynb) top to bottom in Colab. It covers the exact solver, dataset generation, downloading the data and trained models, training curves, the model definition, training (off by default), evaluation, and the inference function.

```python
heights = inference(n, k, m, p_list)   # p_list: list of (k, n-k) arrays -> list of numbers
```

Datasets and trained model files are large and hosted on Google Drive. The notebook downloads them with `gdown`; they are not in this repo.
