import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split

def load_lecun_pt(train_pt="train1989.pt", test_pt="test1989.pt"):
    Xtr, Ytr = torch.load(train_pt)
    Xte, Yte = torch.load(test_pt)
    Xtr = Xtr.view(Xtr.size(0), -1).numpy().astype(np.float64)
    Xte = Xte.view(Xte.size(0), -1).numpy().astype(np.float64)
    ytr = Ytr.argmax(dim=1).numpy()
    yte = Yte.argmax(dim=1).numpy()
    return Xtr, ytr, Xte, yte

# Load the LeCun dataset
X_train, y_train, X_test, y_test = load_lecun_pt()

# Simple data augmentation - add small noise
def augment_data(X, y, noise_factor=0.1, n_augmented=2):
    """Poor man's data augmentation by adding noise"""
    X_aug = []
    y_aug = []
    
    for i in range(n_augmented):
        # Add Gaussian noise
        noise = np.random.normal(0, noise_factor, X.shape)
        X_noisy = X + noise
        X_aug.append(X_noisy)
        y_aug.append(y.copy())
    
    # Combine original and augmented data
    X_combined = np.vstack([X] + X_aug)
    y_combined = np.hstack([y] + y_aug)
    
    return X_combined, y_combined

# Apply data augmentation to training data
X_train_aug, y_train_aug = augment_data(X_train, y_train, noise_factor=0.1, n_augmented=2)

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)

# Use augmented training data
clf.fit(X_train_aug, y_train_aug)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)


print(
    f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)

# Print dataset sizes for comparison
print(f"Original training set size: {X_train.shape[0]}")
print(f"Augmented training set size: {X_train_aug.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()