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
<img width="377" height="398" alt="Screenshot 2025-10-17 at 4 05 03 PM" src="https://github.com/user-attachments/assets/f57fc2c3-f2a5-4c65-a62d-4792954d696c" />
<img width="377" height="398" alt="Screenshot 2025-10-17 at 4 04 27 PM" src="https://github.com/user-attachments/assets/38688cf4-2cee-4bcb-aac5-974f58de018f" />
<img width="377" height="398" alt="Screenshot 2025-10-17 at 4 04 56 PM" src="https://github.com/user-attachments/assets/6cafdd8e-3603-405f-bfce-f7a53012d4da" />


---

## TODO

- [x] **Plot trajectories across time** (e.g., overlay epoch-0, epoch-10, epoch-50)
- [x] **Switch to largest-magnitude eigenvalues**, not just positive ones — to preserve directions with large negative λ (as noted in paper)
- [x] Add explained variance plot (required for interpreting InPCA visuals)

---

## Reference

Quinn, C.J. et al. (2024).  
*Visualizing Probabilistic Models and Data with Intensive Principal Component Analysis.*  
Proceedings of the National Academy of Sciences (PNAS).  
[https://www.pnas.org/doi/10.1073/pnas.1817218116](https://www.pnas.org/doi/10.1073/pnas.1817218116)


https://github.com/user-attachments/assets/8da17c38-ca1c-4c02-b07f-d439590a81ce


---

## Contact

Implemented and modified by norahty.  
For questions, contact directly via email norahty [at] seas.upenn.edu
