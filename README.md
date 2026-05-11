# MNIST PCA-LDA-QDA Classification

This project implements a complete Statistical Machine Learning pipeline for handwritten digit classification using the MNIST dataset. The program classifies digits **0, 1, and 2** using dimensionality reduction, discriminant analysis, and probabilistic classification techniques.

---

## Features

- Loads and preprocesses MNIST handwritten digit data
- Filters and balances classes 0, 1, and 2
- Applies Principal Component Analysis (PCA)
- Reconstructs images from reduced dimensions
- Performs Fisher Discriminant Analysis (FDA)
- Uses Linear Discriminant Analysis (LDA)
- Uses Quadratic Discriminant Analysis (QDA)
- Visualizes PCA and FDA feature projections
- Evaluates classification accuracy under different PCA settings

---

## Algorithms Used

- Principal Component Analysis (PCA)
- Fisher Discriminant Analysis (FDA)
- Linear Discriminant Analysis (LDA)
- Quadratic Discriminant Analysis (QDA)

---

## Dataset

The project uses the MNIST handwritten digit dataset in CSV format.

Classes used:
- 0
- 1
- 2

---

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Required Libraries

```text
numpy
pandas
matplotlib
scikit-learn
```

---

## Running the Project

Run the Python file:

```bash
python sml.py
```

---

## Output

The program:
- Displays reconstructed handwritten digit images
- Shows PCA and FDA scatter plots
- Computes reconstruction MSE
- Prints LDA and QDA training/testing accuracy
- Compares accuracy across PCA variance settings

---

## Project Structure

```text
mnist-pca-lda-qda-classification/
│
├── sml.py
├── README.md
├── requirements.txt
└── sample_outputs/
```

---

## Author

Sanskar
