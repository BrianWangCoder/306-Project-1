# %% [markdown]
# # COMPSYS 306 Project 1 - MLP for traffic sign recognition
#
# Structured like Lab 3 (Steps 10-11, Tasks 10-12), with the Fashion MNIST
# loader replaced by the folder-based loader from Lab 4 Step 5.
#
# Set `DATA_ROOT` in the next cell, then run the cells top to bottom.

# %%
# Importing the necessary libraries
import os
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ---- configuration (every value here goes in the report) ----
DATA_ROOT  = "data/traffic_signs/myData"      # folder holding the class sub-folders
LABELS_CSV = "data/traffic_signs/labels.csv"  # ClassId -> Name lookup
IMG_SIZE   = 32        # images are resized to IMG_SIZE x IMG_SIZE
COLOR_MODE = "L"       # "L" = greyscale (1024 features), "RGB" = colour (3072)
TEST_SIZE  = 0.15      # fraction of the whole dataset held out for testing
VAL_SIZE   = 0.15      # fraction of the whole dataset held out for validation
BATCH_SIZE = 128
EPOCHS     = 30
LR         = 0.001     # Adam learning rate
SEED       = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

# %% [markdown]
# **Step 1: Load the database**
#
# The dataset is a folder of sub-folders, one per class, exactly like the
# `Cars / Ice cream cone / Cricket ball` database in Lab 4. Each image is
# opened, converted, resized to a fixed size and scaled to the range 0-1.
# A fixed size is required because the MLP has a fixed number of input neurons.

# %%
from PIL import Image

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
        # scale the pixel values from the range 0-255 to 0-1 (normalization)
        image_arr.append(np.asarray(img_resized, dtype=np.float32) / 255.0)
        target_arr.append(categories.index(i))

images = np.array(image_arr, dtype=np.float32)
labels = np.array(target_arr, dtype=np.int64)

print("\nThe shape of the image data is", images.shape)
print("The number of labelled samples is", len(labels))
print("The number of classes is", len(categories))
print("Samples per class:", np.bincount(labels))

# %% [markdown]
# **Step 2: Class names**
#
# The folder names are numbers, so `labels.csv` is used to get readable sign
# names for the classification report and the confusion matrix.

# %%
try:
    label_df = pd.read_csv(LABELS_CSV)
    print(label_df.head())
    name_lookup = dict(zip(label_df["ClassId"].astype(str), label_df["Name"]))
    class_names = [name_lookup.get(c, c) for c in categories]
except Exception as e:
    print("Could not read labels.csv, falling back to folder names:", e)
    class_names = list(categories)

print("\nFirst five classes:", class_names[:5])

# %% [markdown]
# **Step 3: Analysis of the data**
#
# Displaying a sample of images with the class name below each one, and the
# class distribution. The dataset is clearly imbalanced, which is why the split
# is stratified later on and why the F1-score matters more than accuracy alone.

# %%
plt.figure(figsize=(10, 10))
for i in range(25):
    index = np.random.randint(len(images))
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images[index], cmap=plt.cm.binary)
    plt.xlabel(class_names[labels[index]][:20], fontsize=7)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.bar(range(len(categories)), np.bincount(labels))
plt.xlabel("Class ID")
plt.ylabel("Number of images")
plt.title("Class distribution")
plt.show()

# %% [markdown]
# **Step 4: Split the database into training, validation and testing sets**
#
# A 70-15-15 split. The validation set is used to monitor generalisation while
# training and to choose the hyperparameters; the testing set is used only once
# at the very end. `stratify` keeps the class proportions the same in all three
# sets, which matters because the classes are imbalanced.

# %%
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=TEST_SIZE, random_state=SEED, stratify=labels)

# VAL_SIZE is a fraction of the whole dataset, so rescale it for this second split
relative_val = VAL_SIZE / (1.0 - TEST_SIZE)
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=relative_val, random_state=SEED,
    stratify=train_labels)

print("Training set  :", train_images.shape, train_labels.shape)
print("Validation set:", val_images.shape, val_labels.shape)
print("Testing set   :", test_images.shape, test_labels.shape)

# %% [markdown]
# **Step 5: Standardization**
#
# Each feature (pixel) is given zero mean and unit variance using the z-score
# from the lecture notes. The mean and standard deviation are computed on the
# **training set only** and then applied to all three sets, so that no
# information from the validation or testing data leaks into training.

# %%
pixel_mean = train_images.mean(axis=0)
pixel_std = train_images.std(axis=0) + 1e-8   # +1e-8 avoids dividing by zero

train_images = (train_images - pixel_mean) / pixel_std
val_images = (val_images - pixel_mean) / pixel_std
test_images = (test_images - pixel_mean) / pixel_std

print("Training set mean after standardization:", train_images.mean().round(4))
print("Training set std  after standardization:", train_images.std().round(4))

# %% [markdown]
# **Step 6: Defining the model structure**
#
# `Flatten` reshapes each IMG_SIZE x IMG_SIZE image into a single vector.
# Two hidden layers with ReLU activation follow, and the output layer has one
# neuron per class. No softmax is applied here because `CrossEntropyLoss`
# applies it internally.

# %%
n_features = IMG_SIZE * IMG_SIZE * (3 if COLOR_MODE == "RGB" else 1)
n_classes = len(categories)

model = nn.Sequential(
    nn.Flatten(),                    # (batch, 32, 32)  -> (batch, 1024)
    nn.Linear(n_features, 512),      # fully connected: 1024 inputs -> 512 outputs
    nn.ReLU(),                       # activation function
    nn.Linear(512, 256),             # 512 inputs -> 256 outputs
    nn.ReLU(),                       # activation function
    nn.Linear(256, n_classes))       # 256 inputs -> one output per class

print(model)
print("\nTotal trainable parameters:",
      sum(p.numel() for p in model.parameters() if p.requires_grad))

# %% [markdown]
# **Step 7: Model settings**
#
# The loss function measures how wrong the model is; the optimizer decides how
# the weights are updated from that loss.

# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Convert the numpy arrays into tensors and wrap the training set in a DataLoader
train_x = torch.tensor(train_images, dtype=torch.float32)
train_y = torch.tensor(train_labels, dtype=torch.long)
train_loader = DataLoader(TensorDataset(train_x, train_y),
                          batch_size=BATCH_SIZE, shuffle=True)

val_x = torch.tensor(val_images, dtype=torch.float32)
val_y = torch.tensor(val_labels, dtype=torch.long)

test_x = torch.tensor(test_images, dtype=torch.float32)
test_y = torch.tensor(test_labels, dtype=torch.long)

# %% [markdown]
# **Step 8: Training**
#
# The training history is recorded each epoch so the accuracy and loss curves
# can be plotted afterwards. The weights from the epoch with the best
# validation accuracy are kept - this is the early stopping idea from the
# lecture notes, which prevents the final model being an overfitted one.

# %%
history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}
best_val_acc = 0.0
best_epoch = 0
best_weights = copy.deepcopy(model.state_dict())

for epoch in range(EPOCHS):
    # ---- training ----
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        correct += (logits.argmax(dim=1) == yb).sum().item()
        total += yb.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    # ---- validation ----
    model.eval()
    with torch.no_grad():
        val_logits = model(val_x)
        val_loss = criterion(val_logits, val_y).item()
        val_acc = (val_logits.argmax(dim=1) == val_y).float().mean().item()

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc, best_epoch = val_acc, epoch + 1
        best_weights = copy.deepcopy(model.state_dict())

    print(f"Epoch {epoch + 1}/{EPOCHS} - loss: {train_loss:.4f}, acc: {train_acc:.4f}, "
          f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

print(f"\nBest validation accuracy {best_val_acc:.4f} at epoch {best_epoch}")
model.load_state_dict(best_weights)   # restore the best model

# %% [markdown]
# **Step 9: Plotting the training history**
#
# Epoch vs accuracy and epoch vs loss, for training and validation. If the
# validation loss starts rising while the training loss keeps falling, the
# model has begun to overfit - that is the point early stopping picks out.

# %%
epochs_range = range(1, len(history['train_loss']) + 1)

plt.figure(figsize=(7, 4))
plt.plot(epochs_range, history['train_acc'], label='Training accuracy')
plt.plot(epochs_range, history['val_acc'], label='Validation accuracy')
plt.title('Epoch vs Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(epochs_range, history['train_loss'], label='Training loss')
plt.plot(epochs_range, history['val_loss'], label='Validation loss')
plt.title('Epoch vs Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
# **Step 10: Evaluating the model on the testing set**

# %%
model.eval()
with torch.no_grad():
    test_logits = model(test_x)
    test_loss = criterion(test_logits, test_y).item()
    predictions = torch.softmax(test_logits, dim=1).numpy()

y_pred = predictions.argmax(axis=1)
y_true = test_labels

print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {accuracy_score(y_true, y_pred):.4f}")

# %% [markdown]
# **Step 11: Classification metrics**
#
# Precision, recall and F1-score per class, plus the macro and weighted
# averages at the bottom of the report. The macro average treats every class
# equally; the weighted average accounts for how many samples each class has.

# %%
print(classification_report(y_true, y_pred, target_names=class_names,
                            digits=4, zero_division=0))

# %% [markdown]
# **Step 12: Confusion matrix**
#
# Rows are the actual class, columns are the predicted class, so the diagonal
# holds the correct classifications. It is normalised by row (divided by the
# number of samples in each true class) so that classes of different sizes can
# be compared.

# %%
cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
cm_normalised = cm / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(12, 10))
plt.imshow(cm_normalised, cmap='Blues')
plt.colorbar(label='Proportion of true class')
plt.xticks(range(n_classes), class_names, rotation=90, fontsize=6)
plt.yticks(range(n_classes), class_names, fontsize=6)
plt.xlabel('Predicted class')
plt.ylabel('Actual class')
plt.title('MLP confusion matrix (testing set)')
plt.tight_layout()
plt.show()

# The most confused pairs - useful discussion material for the report
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
# **Step 13: Looking at individual predictions**
#
# Correct predictions are labelled in blue, incorrect ones in red, with the
# model's confidence in brackets.

# %%
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    color = 'blue' if predicted_label == true_label else 'red'
    plt.xlabel(f"{class_names[predicted_label][:15]} "
               f"{100 * np.max(predictions_array):2.0f}%",
               color=color, fontsize=7)

plt.figure(figsize=(12, 6))
for k in range(18):
    index = np.random.randint(len(test_labels))
    plt.subplot(3, 6, k + 1)
    plot_image(index, predictions[index], test_labels, test_images)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Step 14: Saving the model**
#
# The weights and the standardization statistics are saved together, since new
# images have to be standardized with the same mean and standard deviation
# before the model can be used on them.

# %%
torch.save({'state_dict': model.state_dict(),
            'pixel_mean': pixel_mean,
            'pixel_std': pixel_std,
            'categories': categories,
            'class_names': class_names,
            'img_size': IMG_SIZE,
            'color_mode': COLOR_MODE},
           'mlp_traffic_signs.pt')
print("Model saved to mlp_traffic_signs.pt")

# Record the headline numbers so the SVM notebook can be compared against them
np.save('mlp_predictions.npy', y_pred)
np.save('mlp_true_labels.npy', y_true)
print("Predictions saved for the model comparison")