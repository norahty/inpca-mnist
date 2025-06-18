# InPCA-MNIST (Reproduction and Extension of Figure 3 from Quinn et al.)

This repo reproduces **Figure 3 from _Visualizing Probabilistic Models and Data with Intensive Principal Component Analysis_ (PNAS 2024)** using a TinyCNN trained on MNIST. It also implements the full **intensive InPCA embedding pipeline** as described in Equations (3), (14)–(20) of the paper.

The goal is to track and visualise the prediction geometry of a neural network over training, and identify clustering behavior in the model manifold as it converges.

---

## What this repo contains

- `inpca_mnist.ipynb`: full pipeline training a TinyCNN on MNIST and saving softmax probabilities at specified epochs
- `inpca.py`: implements the intensive InPCA procedure with log-Bhattacharyya distance and double-centering
- `tinycnn.py`: simple 2-layer convolutional network architecture
- `data/`: saved `.npy` files of softmax outputs from test set over training checkpoints

---

## Methods Overview

**InPCA steps** implemented as per Quinn et al.:

1. Apply Hellinger map:                    `Z = √P`
2. Compute overlap matrix:                `H = Z @ Z.T`
3. Apply replica-0 limit (log distance):       `L = 4 log(H)`
4. Double-centering for mean-shift:         `W = J L J`
5. Eigendecomposition of \( W \):          `eigh(W)`
6. Project onto top components:           `coords = U √Σ`

---

## Visuals

Plots show 3D InPCA projections of the model's predictions at different stages of training.  
Clusters emerge as the CNN learns to separate digits; the trajectories replicate the trend in Fig. 3 (Quinn et al.).

---

## TODO

- [x] **Plot trajectories across time** (e.g., overlay epoch-0, epoch-10, epoch-50)
- [x] **Switch to largest-magnitude eigenvalues**, not just positive ones — to preserve directions with large negative λ (as noted in paper)
- [x] Add explained variance plot (required for interpreting InPCA visuals)

- [ ] Optional: support **full 10,000 point** embedding (current default is 2k)

---

## Reference

Quinn, C.J. et al. (2024).  
*Visualizing Probabilistic Models and Data with Intensive Principal Component Analysis.*  
Proceedings of the National Academy of Sciences (PNAS).  
[https://www.pnas.org/doi/10.1073/pnas.1817218116](https://www.pnas.org/doi/10.1073/pnas.1817218116)

---

## Contact

Implemented and modified by norahty.  
For questions, contact directly on slack.
