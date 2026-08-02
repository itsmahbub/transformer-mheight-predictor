# Transformer m-Height Predictor

A neural network that predicts the m-height of a linear code, so you don't have to run the slow exact solver.

Coursework for CSCE 636 (Deep Learning), Texas A&M, Fall 2025. Built with TensorFlow/Keras; the exact solver uses SciPy.

## The problem

A systematic linear code is described by a matrix `G = [I_k | P]`, where `G` has k rows and n columns, and `P` has k rows and n-k columns. Since the identity part `I_k` is always the same, the only thing that varies is `P`.

For a given number m, the m-height of `G` is a single number: take every codeword, sort its entries by absolute value, and divide the largest by the m-th largest. The m-height is the biggest such ratio over all codewords. It is always at least 1.

Computing it exactly means solving a very large number of linear programs, one for each choice of two coordinates, a subset of the remaining coordinates, and a sign pattern. The count grows fast with n and m, so it is slow. `compute_hm` in the notebook is that exact solver, and it is also what produced the training labels.

The goal here is to skip the solver: given `(n, k, m, P)`, predict the m-height directly.

## Approach

Train one model per `(k, n, m)` combination: 21 models in total, covering k in {4, 5, 6}, n in {9, 10}, and m from 2 up to n-k. Separate models make sense because the input size depends on `(k, n)`, and the answers look very different for different m. A single shared model would have to handle both at once.

Three choices that mattered:

- **Predict the log of the m-height, not the raw value.** Raw m-heights span several orders of magnitude and are heavily skewed. Taking the log makes the target well behaved, and the final layer uses ReLU so the prediction can never go below 0 (which corresponds to an m-height of 1, the smallest possible value). The inference function applies `exp` to undo this.
- **Data augmentation by shuffling columns.** Swapping the columns of `G` does not change its m-height, because the answer only depends on sorted absolute values. Each training matrix becomes four samples: the original plus three random column shuffles of `P`.
- **Divide the inputs by 100.** Entries of `P` are drawn uniformly from -100 to 100, while the positional encoding added on top of them ranges from -1 to 1. Without rescaling, the raw entries drown out the positional signal. The same division is applied during training and inference.

### Model

```
Input (k, n-k)
  -> Reshape to (k*(n-k), 1)    # one token per entry of P
  -> PositionalEmbedding        # fixed sinusoidal encoding, added on
  -> TransformerEncoder x 8     # attention + feed-forward, with residuals and layer norm
  -> GlobalMaxPooling1D
  -> Dropout(0.5) -> Dense(128) -> Dropout(0.5)
  -> Dense(1, relu)             # log of the m-height
```

`PositionalEmbedding` and `TransformerEncoder` are custom layers defined in the notebook. They are registered as serializable, so saved models reload without extra setup.

Settings: 2 attention heads, embedding size 64, feed-forward size 32, Adam optimizer, MSE loss, batch size 1024, 30 epochs, 30% validation split. The models are small: 44,489 parameters for k=4, n=9.

Training used a pooled dataset of about 10 million labelled matrices shared across the class, split 70/30 into training and validation. A separate set of 21,000 matrices generated locally (1,000 per configuration) was held out for testing.

## Results

Scores below come from the course evaluator on the held-out test set. Lower is better. Bold marks the hardest case, m = n-k.

| n, k | m=2 | m=3 | m=4 | m=5 | m=6 |
|---|---|---|---|---|---|
| 9, 4 | 0.174 | 0.197 | 0.633 | **3.138** | |
| 10, 4 | 0.842 | 0.084 | 0.304 | 0.823 | **3.267** |
| 9, 5 | 0.228 | 0.589 | **3.502** | | |
| 10, 5 | 0.103 | 0.323 | 0.972 | **2.974** | |
| 9, 6 | 0.537 | **2.748** | | | |
| 10, 6 | 0.135 | 0.632 | **3.453** | | |

Average score: 1.222.

The error is very lopsided. Every configuration with m = n-k scores around 3.0 to 3.5, while everything else stays under 1.0. At the largest m, the m-height distribution grows a long tail of rare very large values, and the log-based model fits the common cases well but misses those extremes. Almost all of the average error comes from those six configurations, so that is where any further work should start.

## Data format

Inputs and outputs are Python lists saved with `pickle.dump()`.

- Input file: a list of samples, each one `[n, k, m, P]`, where `P` is a numpy array with k rows and n-k columns.
- Output file: a list of m-heights (plain numbers), in the same order as the input list.

The provided training set has 32,087 samples with n=9. Test sets use the same format, so a predictions file must be a list of numbers matching the input list one for one.

## Usage

```bash
pip install gdown tamu_csce_636_project1 tensorflow scipy joblib pandas matplotlib
```

Run [the notebook](Project_DL636_Mahbub_Alam.ipynb) top to bottom in Colab. It walks through the exact solver, dataset generation, downloading the data and trained models, training curves, the model definition, training (turned off by default), evaluation, and the inference function.

```python
heights = inference(n, k, m, p_list)   # p_list: list of (k, n-k) arrays -> list of numbers
```

Datasets and trained model files are large and live on Google Drive. The notebook fetches them with `gdown`; they are not stored in this repo.
