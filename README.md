## Tensor Networks for Machine Learning

Hi, this project is based on the paper [Supervised Learning with Tensor Networks](https://papers.nips.cc/paper_files/paper/2016/hash/5314b9674c86e3f9d1ba25ef9bb32895-Abstract.html) by E. M. Stoudenmire and David J. Schwab, 2016. 

This projects contains the following files :

- **[`Embedding.ipynb`](./Embedding.ipynb)** - Preprocess the data to encode every pixel in sin/cos following a snake pattern.
- **[`MPS.py`](./MPS.py)** - Creates a random tensor and puts the MPS in a canonical form.
- **[`Supervised_TN_forML.ipynb`](./Supervised_TN_forML.ipynb)** - Evaluates model performance on test data.
- **[`Sweeper.py`](./Sweeper.py)** - Performs the gradient descent step of the MPS between adjacent tensors. 
