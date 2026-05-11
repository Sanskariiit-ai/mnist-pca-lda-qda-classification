import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# ----------------------------------------------------------
# 1. LOAD MNIST DATA FROM KAGGLE CSV FILES
# ----------------------------------------------------------

train_data = pd.read_csv("mnist_train.csv")
test_data = pd.read_csv("mnist_test.csv")

# ----------------------------------------------------------
# 2. FILTER DIGITS 0,1,2
# ----------------------------------------------------------

train_data = train_data[train_data["label"].isin([0,1,2])]
test_data = test_data[test_data["label"].isin([0,1,2])]

train_data = train_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)

# ----------------------------------------------------------
# 3. SAMPLE 100 TRAIN + 100 TEST PER CLASS
# ----------------------------------------------------------

train_sample = train_data.groupby("label").sample(100, random_state=42)
test_sample = test_data.groupby("label").sample(100, random_state=42)

train_sample = train_sample.reset_index(drop=True)
test_sample = test_sample.reset_index(drop=True)

# ----------------------------------------------------------
# 4. CREATE FEATURE MATRICES
# ----------------------------------------------------------

X_train = train_sample.drop("label", axis=1).values / 255.0
y_train = train_sample["label"].values

X_test = test_sample.drop("label", axis=1).values / 255.0
y_test = test_sample["label"].values

# transpose to match assignment X ∈ R^(784 × 300)
X_train = X_train.T
X_test = X_test.T

# ----------------------------------------------------------
# 5. PCA IMPLEMENTATION
# ----------------------------------------------------------

# compute mean
mu = np.mean(X_train, axis=1, keepdims=True)

# mean center
Xc = X_train - mu

# covariance matrix
S = (Xc @ Xc.T) / (Xc.shape[1] - 1)

# eigen decomposition
eigvals, eigvecs = np.linalg.eigh(S)

# sort eigenvalues descending
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

# cumulative variance
var_ratio = eigvals / np.sum(eigvals)
cum_var = np.cumsum(var_ratio)

# retain 75% variance
p = np.where(cum_var >= 0.75)[0][0]

Up = eigvecs[:, :p]

# project training data
Y_train = Up.T @ Xc

# ----------------------------------------------------------
# 6. PROJECT TEST DATA
# ----------------------------------------------------------

X_test_centered = X_test - mu
Y_test = Up.T @ X_test_centered

# ----------------------------------------------------------
# 7. RECONSTRUCTION
# ----------------------------------------------------------

X_reconstructed = Up @ Y_train + mu

mse = np.mean((X_train[:, :5] - X_reconstructed[:, :5])**2)

print("Reconstruction MSE:", mse)

for i in range(5):

    plt.figure(figsize=(6,3))

    plt.subplot(1,2,1)
    plt.imshow(X_train[:,i].reshape(28,28), cmap="gray")
    plt.title("Original")

    plt.subplot(1,2,2)
    plt.imshow(X_reconstructed[:,i].reshape(28,28), cmap="gray")
    plt.title("Reconstructed")

    plt.show()

# ----------------------------------------------------------
# 8. FISHER DISCRIMINANT ANALYSIS
# ----------------------------------------------------------

classes = [0,1,2]

mean_total = np.mean(Y_train, axis=1, keepdims=True)

dim = Y_train.shape[0]

SB = np.zeros((dim, dim))
SW = np.zeros((dim, dim))

for c in classes:

    Xc_class = Y_train[:, y_train == c]

    mean_c = np.mean(Xc_class, axis=1, keepdims=True)

    Nc = Xc_class.shape[1]

    SB += Nc * (mean_c - mean_total) @ (mean_c - mean_total).T

    for i in range(Nc):

        diff = Xc_class[:, i:i+1] - mean_c
        SW += diff @ diff.T

# generalized eigenvalue problem
eigvals_fda, eigvecs_fda = np.linalg.eig(np.linalg.inv(SW) @ SB)

idx = np.argsort(eigvals_fda)[::-1]

W = eigvecs_fda[:, idx[:2]]

Z_train = W.T @ Y_train
Z_test = W.T @ Y_test

# ----------------------------------------------------------
# 9. LDA CLASSIFIER
# ----------------------------------------------------------

lda = LinearDiscriminantAnalysis()

lda.fit(Y_train.T, y_train)

train_pred_lda = lda.predict(Y_train.T)
test_pred_lda = lda.predict(Y_test.T)

print("LDA Train Accuracy:", np.mean(train_pred_lda == y_train))
print("LDA Test Accuracy:", np.mean(test_pred_lda == y_test))

# ----------------------------------------------------------
# 10. QDA CLASSIFIER
# ----------------------------------------------------------

qda = QuadraticDiscriminantAnalysis()

qda.fit(Y_train.T, y_train)

train_pred_qda = qda.predict(Y_train.T)
test_pred_qda = qda.predict(Y_test.T)

print("QDA Train Accuracy:", np.mean(train_pred_qda == y_train))
print("QDA Test Accuracy:", np.mean(test_pred_qda == y_test))

# ----------------------------------------------------------
# 11. PCA VISUALIZATION
# ----------------------------------------------------------

plt.figure()

plt.scatter(Y_train[0,:], Y_train[1,:], c=y_train)

plt.title("PCA Projection")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.show()

# ----------------------------------------------------------
# 12. FDA VISUALIZATION
# ----------------------------------------------------------

plt.figure()

plt.scatter(Z_train[0,:], Z_train[1,:], c=y_train)

plt.title("FDA Projection")
plt.xlabel("FD1")
plt.ylabel("FD2")

plt.show()

# ----------------------------------------------------------
# 13. PCA WITH 90% VARIANCE
# ----------------------------------------------------------

p90 = np.where(cum_var >= 0.90)[0][0]

Up90 = eigvecs[:, :p90]

Y_train90 = Up90.T @ Xc
Y_test90 = Up90.T @ X_test_centered

lda90 = LinearDiscriminantAnalysis()

lda90.fit(Y_train90.T, y_train)

acc90 = np.mean(lda90.predict(Y_test90.T) == y_test)

print("Accuracy with 90% PCA variance:", acc90)

# ----------------------------------------------------------
# 14. USING ONLY FIRST 2 PRINCIPAL COMPONENTS
# ----------------------------------------------------------

Up2 = eigvecs[:, :2]

Y_train2 = Up2.T @ Xc
Y_test2 = Up2.T @ X_test_centered

lda2 = LinearDiscriminantAnalysis()

lda2.fit(Y_train2.T, y_train)

train_acc2 = np.mean(lda2.predict(Y_train2.T) == y_train)
test_acc2 = np.mean(lda2.predict(Y_test2.T) == y_test)

print("Train accuracy with 2 PCs:", train_acc2)
print("Test accuracy with 2 PCs:", test_acc2)