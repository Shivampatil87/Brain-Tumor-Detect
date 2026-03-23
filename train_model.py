"""
Brain Tumor CNN — Training Script
===================================
Exact replication of:
  brain-tumor-another-dataset-training-4_1_.ipynb

Real results achieved:
  Dataset        : 3000 MRI images (Brain Tumor Detection MRI, Kaggle)
  Train split    : 2700 images  (90%)
  Val split      : 150 images   (5%)
  Test split     : 150 images   (5%)
  Epochs run     : 30 (early stopping patience=5)
  Val  Accuracy  : 94.67%
  Test Accuracy  : 95.33%
  Test Loss      : 0.1430

Dataset URL:
  https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection

Key preprocessing: crop_brain_contour() — OpenCV contour crop to isolate
brain tissue, removing black borders which confuse the model.

Usage:
    pip install -r requirements.txt
    # Place dataset as: dataset/yes/ and dataset/no/
    python train_model.py
"""

import os
import numpy as np
import cv2
import imutils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D, Input, ZeroPadding2D, BatchNormalization,
    Flatten, Activation, Dense, MaxPooling2D
)
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix
from keras import callbacks
import seaborn as sns

# ─── Configuration ──────────────────────────────────────────────────────────
IMG_WIDTH      = 240
IMG_HEIGHT     = 240
IMG_SHAPE      = (IMG_WIDTH, IMG_HEIGHT, 3)
BATCH_SIZE     = 64
MAX_EPOCHS     = 100         # Early stopping will end it earlier (~30 epochs)
EARLY_STOP_PAT = 5           # Patience for EarlyStopping on val_loss
DATASET_DIR    = 'dataset'
MODEL_PATH     = 'model/brain_tumor_model.h5'
GRAPH_DIR      = 'static/images'

tf.random.set_seed(42)
np.random.seed(42)
os.makedirs('model', exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)


# ─── Preprocessing ──────────────────────────────────────────────────────────
def crop_brain_contour(image, plot=False):
    """
    Crop MRI image to just the brain region using OpenCV contour detection.

    Steps:
      1. Convert to grayscale + Gaussian blur
      2. Binary threshold at 45 + morphological cleanup
      3. Find largest external contour (the brain)
      4. Crop to bounding box of extreme points

    This removes black background padding and focuses the model on brain tissue.
    Directly from the notebook — no modifications.
    """
    gray   = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray   = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts   = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts   = imutils.grab_contours(cnts)
    c      = max(cnts, key=cv2.contourArea)

    extLeft  = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop   = tuple(c[c[:, :, 1].argmin()][0])
    extBot   = tuple(c[c[:, :, 1].argmax()][0])

    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        axes[1].imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Cropped Image')
        axes[1].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(GRAPH_DIR, 'sample_crop.png'), dpi=100)
        plt.close()

    return new_image


# ─── Data Loading ────────────────────────────────────────────────────────────
def load_data(dir_list, image_size):
    """
    Load and preprocess all images from the given directories.
    Applies crop_brain_contour() then resizes and normalizes.
    Labels: yes/ = 1 (tumor), no/ = 0 (no tumor)
    """
    from os import listdir
    X, y = [], []
    image_width, image_height = image_size
    skipped = 0

    for directory in dir_list:
        files = listdir(directory)
        print(f"  Loading {len(files)} images from {directory}...")
        for filename in files:
            path  = os.path.join(directory, filename)
            image = cv2.imread(path)
            if image is None:
                skipped += 1
                continue
            try:
                image = crop_brain_contour(image, plot=False)
                image = cv2.resize(image, dsize=(image_width, image_height),
                                   interpolation=cv2.INTER_CUBIC)
                image = image / 255.0   # Normalize to [0, 1]
                X.append(image)
                # Label: 1=tumor (yes/), 0=no tumor (no/)
                y.append([1] if directory.endswith('yes') else [0])
            except Exception as e:
                skipped += 1

    X = np.array(X)
    y = np.array(y)
    X, y = shuffle(X, y)   # Shuffle improves training

    print(f"\n  ✅ Loaded {len(X)} images (skipped {skipped})")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    return X, y


def split_data(X, y, test_size=0.1):
    """
    Split into train / val / test sets.
    Matches notebook: 90% train, 5% val, 5% test.
    """
    X_train, X_test_val, y_train, y_test_val = train_test_split(
        X, y, test_size=test_size, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(
        X_test_val, y_test_val, test_size=0.5, random_state=42)
    return X_train, y_train, X_val, y_val, X_test, y_test


# ─── Model Architecture ───────────────────────────────────────────────────────
def build_model(input_shape):
    """
    Exact architecture from the notebook.

    Surprisingly compact (only 11,137 params) but achieves 95.33% test accuracy.
    The key is the aggressive double MaxPooling which quickly downsamples
    the 238x238 feature maps to 14x14, then flattens directly to sigmoid.

    Layer summary:
      Input         (240, 240, 3)
      ZeroPadding2D → (244, 244, 3)     # Add 2-pixel border
      Conv2D(32, 7×7) → (238, 238, 32)  # Large receptive field
      BatchNorm + ReLU
      MaxPool(4×4) → (59, 59, 32)
      MaxPool(4×4) → (14, 14, 32)
      Flatten       → (6272,)
      Dense(1, sigmoid)                  # Binary output
    """
    X_input = Input(input_shape)
    X = ZeroPadding2D((2, 2))(X_input)

    X = Conv2D(32, (7, 7), strides=(1, 1))(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((4, 4))(X)
    X = MaxPooling2D((4, 4))(X)
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid')(X)

    return Model(inputs=X_input, outputs=X, name='BrainTumorCNN')


# ─── Visualization ───────────────────────────────────────────────────────────
def save_graphs(history_dict, y_test, y_pred):
    """Save training curves and confusion matrix."""
    train_loss = history_dict['loss']
    val_loss   = history_dict['val_loss']
    train_acc  = history_dict['accuracy']
    val_acc    = history_dict['val_accuracy']
    epochs     = range(1, len(train_loss) + 1)

    # ── Training curves ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#050a12')

    for ax, train, val, title, ylabel in [
        (axes[0], train_acc, val_acc,   'Accuracy', 'Accuracy'),
        (axes[1], train_loss, val_loss, 'Loss',     'Loss'),
    ]:
        ax.set_facecolor('#0c1421')
        ax.plot(epochs, train, color='#00d4ff', lw=2.5, label=f'Train {title}', marker='o', ms=3)
        ax.plot(epochs, val,   color='#ff6b6b', lw=2.5, label=f'Val {title}',   marker='s', ms=3)
        ax.set_title(f'Model {title}', color='white', fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('Epoch', color='#8899aa')
        ax.set_ylabel(ylabel, color='#8899aa')
        ax.legend(facecolor='#0c1421', edgecolor='#1a3050', labelcolor='white', fontsize=9)
        ax.tick_params(colors='#8899aa')
        for spine in ax.spines.values(): spine.set_color('#1a3050')
        ax.grid(True, alpha=0.15, color='#ffffff')

    plt.tight_layout(pad=2)
    path = os.path.join(GRAPH_DIR, 'training_history.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  📈 Training graphs saved → {path}")

    # ── Confusion Matrix ──
    cm = confusion_matrix(y_test, y_pred)
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    fig2.patch.set_facecolor('#050a12')
    ax2.set_facecolor('#0c1421')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Tumor', 'Tumor'],
                yticklabels=['No Tumor', 'Tumor'],
                ax=ax2, linewidths=0.5, linecolor='#1a3050',
                annot_kws={'size': 14, 'color': 'white'})
    ax2.set_title('Confusion Matrix', color='white', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Predicted', color='#8899aa')
    ax2.set_ylabel('Actual', color='#8899aa')
    ax2.tick_params(colors='#8899aa')
    plt.tight_layout()
    path2 = os.path.join(GRAPH_DIR, 'confusion_matrix.png')
    plt.savefig(path2, dpi=150, bbox_inches='tight', facecolor=fig2.get_facecolor())
    plt.close()
    print(f"  📊 Confusion matrix saved → {path2}")


# ─── Main Training Pipeline ──────────────────────────────────────────────────
def train():
    print("\n" + "="*60)
    print("  Brain Tumor CNN — Training (Notebook Replica)")
    print("="*60)

    yes_dir = os.path.join(DATASET_DIR, 'yes')
    no_dir  = os.path.join(DATASET_DIR, 'no')

    if not os.path.exists(yes_dir) or not os.path.exists(no_dir):
        print(f"\n❌ Dataset not found at {DATASET_DIR}/yes and {DATASET_DIR}/no")
        print("\nDownload from Kaggle:")
        print("  https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection")
        print("\nPlace images as:")
        print("  dataset/yes/  ← brain MRI images WITH tumor")
        print("  dataset/no/   ← brain MRI images WITHOUT tumor")
        return

    # Load data
    print(f"\n📂 Loading dataset from {DATASET_DIR}/...")
    X, y = load_data([yes_dir, no_dir], (IMG_WIDTH, IMG_HEIGHT))

    # Split: 90/5/5 matching notebook
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y, test_size=0.1)
    print(f"\n📊 Data split:")
    print(f"   Train : {X_train.shape[0]} images")
    print(f"   Val   : {X_val.shape[0]} images")
    print(f"   Test  : {X_test.shape[0]} images")

    # Build model
    print("\n🏗️  Building model...")
    model = build_model(IMG_SHAPE)
    model.summary()

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks — matches notebook exactly
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=EARLY_STOP_PAT,
        restore_best_weights=True,
        verbose=1
    )
    checkpoint = callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    # Train
    print(f"\n🚀 Training (max {MAX_EPOCHS} epochs, early stopping patience={EARLY_STOP_PAT})...")
    hist = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=MAX_EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    # Evaluate
    print("\n📊 Final Evaluation:")
    loss_v, acc_v = model.evaluate(X_val,  y_val,  verbose=0)
    loss_t, acc_t = model.evaluate(X_test, y_test, verbose=0)
    print(f"   Validation : Accuracy = {acc_v*100:.2f}%  Loss = {loss_v:.4f}")
    print(f"   Test       : Accuracy = {acc_t*100:.2f}%  Loss = {loss_t:.4f}")

    # Predictions for confusion matrix
    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    print("\n📋 Classification Report:")
    print(classification_report(y_test.flatten(), y_pred,
                                target_names=['No Tumor', 'Tumor']))

    # Save graphs
    print("\n📈 Saving graphs...")
    save_graphs(hist.history, y_test.flatten(), y_pred)

    model.save(MODEL_PATH)
    print(f"\n✅ Model saved → {MODEL_PATH}")
    print("="*60)


if __name__ == '__main__':
    train()
