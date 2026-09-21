# %% [markdown]
# # COMPSYS 306 Project 1 - SVM for traffic sign recognition (no PCA)
#
# Structured like Lab 4 (Steps 3-5, Tasks 4-8): the same folder loading as the
# MLP notebook, a comparison of the linear / polynomial / RBF kernels, a grid
# search over the hyperparameters, then a confusion matrix with the evaluation
# metrics calculated manually.
#
# This version trains the SVM directly on all 1024 standardised pixel features,
# with no dimensionality reduction, so both models see an identical feature
# vector. Steps 1-5 are identical to the MLP notebook on purpose - the same seed
# gives the same split, so the two models are compared on exactly the same data.
#
# **Runtime:** roughly 10x slower than the PCA version, since SVM kernel
# evaluations scale with the number of features. Expect a few minutes for the
# grid search and under ten minutes for the final fit.

# %%
# Importing the necessary libraries
import os
import pickle
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ---- configuration (keep these identical to the MLP notebook) ----
DATA_ROOT  = "data/traffic_signs/myData"
LABELS_CSV = "data/traffic_signs/labels.csv"
IMG_SIZE   = 32
COLOR_MODE = "L"
TEST_SIZE  = 0.15
VAL_SIZE   = 0.15
SEED       = 42

# ---- SVM-specific settings ----
SEARCH_SUBSET = 5000   # samples used for the grid search (0 = use all of them)

np.random.seed(SEED)

# %% [markdown]
# **Step 1: Load the database**
#
# Identical to Lab 4 Step 5: every image is opened, resized to a fixed size and
# scaled to the range 0-1, with the folder name giving the class label.

# %%
categories = sorted(os.listdir(DATA_ROOT),
                    key=lambda name: int(name) if name.isdigit() else name)
categories = [c for c in categories if os.path.isdir(os.path.join(DATA_ROOT, c))]

image_arr = []
target_arr = []

for i in categories:
    print(f"loading... category : {i}")
    path = os.path.join(DATA_ROOT, i)
    for img in os.listdir(path):
        img_array = Image.open(os.path.join(path, img)).convert(COLOR_MODE)
        img_resized = img_array.resize((IMG_SIZE, IMG_SIZE))
        image_arr.append(np.asarray(img_resized, dtype=np.float32) / 255.0)
        target_arr.append(categories.index(i))

images = np.array(image_arr, dtype=np.float32)
labels = np.array(target_arr, dtype=np.int64)

print("\nThe shape of the image data is", images.shape)
print("The number of classes is", len(categories))

# %% [markdown]
# **Step 2: Class names from labels.csv**

# %%
try:
    label_df = pd.read_csv(LABELS_CSV)
    name_lookup = dict(zip(label_df["ClassId"].astype(str), label_df["Name"]))
    class_names = [name_lookup.get(c, c) for c in categories]
except Exception as e:
    print("Could not read labels.csv, falling back to folder names:", e)
    class_names = list(categories)

n_classes = len(categories)
print("First five classes:", class_names[:5])

# %% [markdown]
# **Step 3: Flatten the images**
#
# An SVM takes one feature vector per sample, so each IMG_SIZE x IMG_SIZE image
# becomes a single row of pixel values - the same `.flatten()` step used on the
# Cars / Ice cream cone / Cricket ball database in Lab 4. (The MLP does this
# inside the network with `nn.Flatten()`; here it has to be done beforehand.)

# %%
X = images.reshape(len(images), -1)
y = labels
print("Feature matrix:", X.shape)

# %% [markdown]
# **Step 4: Split the database**
#
# The same 70-15-15 stratified split as the MLP notebook. Because `random_state`
# and the data order are the same, the three sets contain exactly the same
# images in both notebooks.

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)

relative_val = VAL_SIZE / (1.0 - TEST_SIZE)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=relative_val, random_state=SEED, stratify=y_train)

print('Splitted Successfully')
print("Training set  :", X_train.shape)
print("Validation set:", X_val.shape)
print("Testing set   :", X_test.shape)

# %% [markdown]
# **Step 5: Standardization**
#
# `StandardScaler` gives every feature zero mean and unit variance, which is the
# z-score from the lecture notes and the same scaling used on the rice database
# in Lab 4 Tasks 4-6. It matters more for the SVM than for most models because
# the RBF kernel is a distance measure - unscaled features with large ranges
# would dominate the distance.
#
# The scaler is fitted on the training set only, then applied to the validation
# and testing sets, so no information from them leaks into training.

# %%
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print("Training set mean after scaling:", X_train.mean().round(4))
print("Training set std  after scaling:", X_train.std().round(4))

# %% [markdown]
# **Step 6: Feature dimensionality**
#
# No dimensionality reduction is applied: the SVM is trained on all
# 32 x 32 = 1024 standardised pixel features, exactly the same input the MLP
# receives. This keeps the two models directly comparable, at the cost of
# training time - an SVM's kernel evaluations scale with the number of features,
# so training takes roughly ten times longer than it would on 100 components.
#
# One useful consequence of standardising and *not* projecting: every feature
# now has unit variance, so scikit-learn's default `gamma='scale'` evaluates to
# 1 / (1024 x 1) which is approximately 0.001 - a sensible value. The grid below
# brackets it.

# %%
n_features = X_train.shape[1]
print(f"Training the SVM on all {n_features} features (no reduction)")
print(f"Default gamma='scale' would be approximately "
      f"{1 / (n_features * X_train.var()):.5f}")

# %% [markdown]
# **Step 7: Comparing the three kernels (Lab 4 Tasks 4-6)**
#
# Each kernel is trained with the same C so the only difference is the shape of
# the decision boundary: linear gives a straight hyperplane, polynomial a curved
# one, and RBF a flexible local boundary. Accuracy is measured on the validation
# set, never the testing set.
#
# A subset is used here to keep the runtime reasonable - an SVM's training time
# grows roughly quadratically with the number of samples.

# %%
if SEARCH_SUBSET and SEARCH_SUBSET < len(X_train):
    subset_idx = np.random.choice(len(X_train), SEARCH_SUBSET, replace=False)
    X_sub, y_sub = X_train[subset_idx], y_train[subset_idx]
else:
    X_sub, y_sub = X_train, y_train
print(f"Using {len(X_sub)} training samples for kernel comparison and grid search\n")

kernel_results = {}

for kernel_name in ['linear', 'poly', 'rbf']:
    start = time.time()
    clf = svm.SVC(kernel=kernel_name, C=1)
    clf.fit(X_sub, y_sub)
    acc = accuracy_score(y_val, clf.predict(X_val))
    elapsed = time.time() - start
    kernel_results[kernel_name] = acc
    print(f'{kernel_name.capitalize()} kernel accuracy: {acc*100:.2f}%  '
          f'({elapsed:.1f}s, {int(clf.n_support_.sum())} support vectors)')

plt.figure(figsize=(5, 4))
plt.bar(kernel_results.keys(), [v * 100 for v in kernel_results.values()])
plt.ylabel('Validation accuracy (%)')
plt.title('SVM kernel comparison (C = 1)')
plt.show()

# %% [markdown]
# **Step 8: Hyperparameter tuning with GridSearchCV**
#
# Grid search performs an exhaustive search over the specified parameter values,
# as introduced in Lab 4 Step 5. `cv=3` means 3-fold cross-validation: the
# training subset is split into three folds and each takes a turn as the
# validation fold, so the score is averaged over three runs (Topic 7).
#
# * **C** is the regularisation parameter. A small C allows more slack variables
#   (a wider margin, more training errors tolerated); a large C forces the model
#   to classify the training data correctly, risking overfitting.
# * **gamma** controls how far the influence of a single training sample reaches
#   in the RBF kernel. A small gamma gives a smooth boundary, a large gamma a
#   tight one that can overfit. It does not apply to the linear kernel.
#
# The linear kernel is included in the grid so that all three kernels are scored
# under identical conditions, rather than only in the fixed-C comparison above.
#
# This is the slowest cell in the notebook.

# %%
param_grid = [
    {'kernel': ['rbf'],    'C': [0.1, 1, 10], 'gamma': [0.0001, 0.001, 0.01]},
    {'kernel': ['poly'],   'C': [0.1, 1, 10], 'gamma': [0.0001, 0.001, 0.01]},
    {'kernel': ['linear'], 'C': [0.1, 1, 10]},
]

svc = svm.SVC()
print("The training of the model is started, please wait for a while as it may "
      "take several minutes to complete")

start = time.time()
model = GridSearchCV(svc, param_grid, cv=3, n_jobs=-1, verbose=2)
model.fit(X_sub, y_sub)
print(f'\nThe model is trained. Grid search took {time.time() - start:.1f}s')
print('Best parameters:', model.best_params_)
print(f'Best cross-validation accuracy: {model.best_score_*100:.2f}%')

results_df = pd.DataFrame(model.cv_results_)[
    ['param_C', 'param_gamma', 'param_kernel', 'mean_test_score', 'std_test_score']
].sort_values('mean_test_score', ascending=False)
print()
print(results_df.head(10).to_string(index=False))

# %% [markdown]
# **Step 9: Checking the tuned model on the validation set**
#
# The chosen parameters are confirmed on the held-out validation set - the same
# set the MLP was tuned against - before the final model is built.

# %%
val_acc = accuracy_score(y_val, model.predict(X_val))
print(f'Validation accuracy of the tuned model: {val_acc*100:.2f}%')

# %% [markdown]
# **Step 10: Training the final model**
#
# The winning parameters are retrained on the training **and** validation data
# together, so the final model uses as much data as possible. The testing set is
# still untouched at this point. This is the longest single operation in the
# notebook - expect several minutes.

# %%
X_full = np.vstack([X_train, X_val])
y_full = np.concatenate([y_train, y_val])

final_model = svm.SVC(**model.best_params_)

start = time.time()
final_model.fit(X_full, y_full)
train_time = time.time() - start

print(f'Final model trained on {len(y_full)} samples in {train_time:.1f}s')
print(f'Number of support vectors: {int(final_model.n_support_.sum())} '
      f'({final_model.n_support_.sum()/len(y_full):.1%} of the training data)')

# %% [markdown]
# **Step 11: Prediction and accuracy on the testing set**

# %%
start = time.time()
y_pred = final_model.predict(X_test)
predict_time = time.time() - start

print("The predicted data is :", y_pred[:20])
print("The actual data is    :", np.array(y_test[:20]))
print(f"\nThe model is {accuracy_score(y_pred, y_test)*100:.2f}% accurate")
print(f"Prediction of {len(y_test)} samples took {predict_time:.1f}s "
      f"({1000*predict_time/len(y_test):.2f} ms per image)")

# %% [markdown]
# **Step 12: Classification report**

# %%
print(classification_report(y_test, y_pred, target_names=class_names,
                            digits=4, zero_division=0))

# %% [markdown]
# **Step 13: Confusion matrix**
#
# Rows are the actual class and columns the predicted class, so the diagonal
# holds the correct classifications and everything off the diagonal is a
# misclassification. It is normalised by row so classes of different sizes can
# be compared.

# %%
cm = confusion_matrix(y_test, y_pred, labels=np.arange(n_classes))
cm_normalised = cm / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(12, 10))
plt.imshow(cm_normalised, cmap='Blues')
plt.colorbar(label='Proportion of true class')
plt.xticks(range(n_classes), class_names, rotation=90, fontsize=6)
plt.yticks(range(n_classes), class_names, fontsize=6)
plt.xlabel('Predicted class')
plt.ylabel('Actual class')
plt.title('SVM confusion matrix (testing set)')
plt.tight_layout()
plt.show()

confusions = []
for i in range(n_classes):
    for j in range(n_classes):
        if i != j and cm[i, j] > 0:
            confusions.append((cm[i, j], class_names[i], class_names[j]))
confusions.sort(reverse=True)

print("Most common misclassifications (count, actual -> predicted):")
for count, actual, predicted in confusions[:10]:
    print(f"  {count:>4}  {actual}  ->  {predicted}")

# %% [markdown]
# **Step 14: Calculating the metrics manually (Lab 4 Tasks 7-8)**
#
# In Lab 4 the problem was binary, so the confusion matrix gave TP, TN, FP and
# FN directly. With 43 classes the same idea is applied one class at a time
# (the one-vs-all view from Topic 6): for a chosen class,
#
# * **TP** = samples of that class predicted as that class (the diagonal entry)
# * **FN** = samples of that class predicted as something else (rest of the row)
# * **FP** = samples of other classes predicted as that class (rest of the column)
# * **TN** = everything else
#
# Averaging the per-class scores gives the macro average, which is what the
# classification report prints.

# %%
example_class = int(np.argmax(cm.sum(axis=1)))   # the largest class
TP = cm[example_class, example_class]
FN = cm[example_class, :].sum() - TP
FP = cm[:, example_class].sum() - TP
TN = cm.sum() - TP - FN - FP

print(f"Worked example for class '{class_names[example_class]}':")
print(f"  TP (class predicted as itself)         = {TP}")
print(f"  FN (class predicted as something else) = {FN}")
print(f"  FP (something else predicted as class) = {FP}")
print(f"  TN (everything else)                   = {TN}\n")

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"  Accuracy  = (TP+TN)/(TP+TN+FP+FN) = ({TP}+{TN})/{TP+TN+FP+FN} = {accuracy:.4f}")
print(f"  Precision = TP/(TP+FP) = {TP}/{TP+FP} = {precision:.4f}")
print(f"  Recall    = TP/(TP+FN) = {TP}/{TP+FN} = {recall:.4f}")
print(f"  F1 score  = 2*P*R/(P+R) = {f1:.4f}")

precisions, recalls, f1s = [], [], []
for k in range(n_classes):
    tp = cm[k, k]
    fn = cm[k, :].sum() - tp
    fp = cm[:, k].sum() - tp
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    precisions.append(p)
    recalls.append(r)
    f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0)

print(f"\nManually calculated macro averages over all {n_classes} classes:")
print(f"  Overall accuracy   = {np.trace(cm) / cm.sum():.4f}")
print(f"  Precision (macro)  = {np.mean(precisions):.4f}")
print(f"  Recall    (macro)  = {np.mean(recalls):.4f}")
print(f"  F1-score  (macro)  = {np.mean(f1s):.4f}")

print("\nsklearn accuracy check:", accuracy_score(y_test, y_pred))

# %% [markdown]
# **Step 15: Saving the model with pickle**
#
# The scaler is saved alongside the model, because a new image must be scaled
# with exactly the same mean and standard deviation before the model can
# classify it.

# %%
pickle.dump({'model': final_model, 'scaler': scaler,
             'categories': categories, 'class_names': class_names,
             'img_size': IMG_SIZE, 'color_mode': COLOR_MODE},
            open('svm_traffic_signs.p', 'wb'))
print("Pickle is dumped successfully")

saved = pickle.load(open('svm_traffic_signs.p', 'rb'))
print("Reloaded model:", saved['model'])

np.save('svm_predictions.npy', y_pred)
np.save('svm_true_labels.npy', y_test)

# %% [markdown]
# **Step 16: Comparing the MLP and the SVM**
#
# Run the MLP notebook first so `mlp_predictions.npy` exists. Both models were
# trained and evaluated on identical splits with identical features, so these
# numbers are directly comparable - this table is the comparative analysis the
# report asks for.

# %%
from sklearn.metrics import precision_score, recall_score, f1_score

try:
    mlp_pred = np.load('mlp_predictions.npy')
    mlp_true = np.load('mlp_true_labels.npy')
    assert np.array_equal(mlp_true, y_test), \
        "The two notebooks used different splits - check SEED and the config values"

    comparison = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision (macro)', 'Recall (macro)',
                   'F1-score (macro)', 'F1-score (weighted)'],
        'MLP': [accuracy_score(y_test, mlp_pred),
                precision_score(y_test, mlp_pred, average='macro', zero_division=0),
                recall_score(y_test, mlp_pred, average='macro', zero_division=0),
                f1_score(y_test, mlp_pred, average='macro', zero_division=0),
                f1_score(y_test, mlp_pred, average='weighted', zero_division=0)],
        'SVM': [accuracy_score(y_test, y_pred),
                precision_score(y_test, y_pred, average='macro', zero_division=0),
                recall_score(y_test, y_pred, average='macro', zero_division=0),
                f1_score(y_test, y_pred, average='macro', zero_division=0),
                f1_score(y_test, y_pred, average='weighted', zero_division=0)],
    })
    print(comparison.round(4).to_string(index=False))
    comparison.round(4).to_csv('model_comparison.csv', index=False)

    both_right = ((mlp_pred == y_test) & (y_pred == y_test)).sum()
    only_mlp = ((mlp_pred == y_test) & (y_pred != y_test)).sum()
    only_svm = ((mlp_pred != y_test) & (y_pred == y_test)).sum()
    both_wrong = ((mlp_pred != y_test) & (y_pred != y_test)).sum()
    print(f"\nOut of {len(y_test)} testing samples:")
    print(f"  both correct      : {both_right}")
    print(f"  only the MLP right: {only_mlp}")
    print(f"  only the SVM right: {only_svm}")
    print(f"  both wrong        : {both_wrong}")

except FileNotFoundError:
    print("mlp_predictions.npy not found - run the MLP notebook first.")