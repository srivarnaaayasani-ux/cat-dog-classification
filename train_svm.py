import os
import cv2
import joblib
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ==========================================
# SETTINGS
# ==========================================

CAT_FOLDER = "data/cats"
DOG_FOLDER = "data/dogs"

IMAGE_SIZE = (128, 128)

MODEL_FOLDER = "models"
IMAGE_FOLDER = "images"

os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)


# ==========================================
# HOG FEATURE EXTRACTION
# ==========================================

def extract_hog_features(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return None

    # Resize image
    image = cv2.resize(image, IMAGE_SIZE)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Extract HOG features
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    return features


# ==========================================
# LOAD DATASET
# ==========================================

print("\nLoading dataset...")

features = []
labels = []

# ------------------------------------------
# LOAD CATS
# ------------------------------------------

print("Processing cat images...")

cat_files = os.listdir(CAT_FOLDER)

for i, file in enumerate(cat_files):

    image_path = os.path.join(CAT_FOLDER, file)

    feature = extract_hog_features(image_path)

    if feature is not None:
        features.append(feature)
        labels.append(0)

    if (i + 1) % 500 == 0:
        print(f"Processed {i + 1} cat images")


# ------------------------------------------
# LOAD DOGS
# ------------------------------------------

print("\nProcessing dog images...")

dog_files = os.listdir(DOG_FOLDER)

for i, file in enumerate(dog_files):

    image_path = os.path.join(DOG_FOLDER, file)

    feature = extract_hog_features(image_path)

    if feature is not None:
        features.append(feature)
        labels.append(1)

    if (i + 1) % 500 == 0:
        print(f"Processed {i + 1} dog images")


# ==========================================
# CONVERT TO NUMPY
# ==========================================

X = np.array(features)
y = np.array(labels)

print("\nDataset loaded successfully!")

print("Total images:", len(X))
print("Feature size:", X.shape[1])

print("Cat images:", np.sum(y == 0))
print("Dog images:", np.sum(y == 1))


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining images:", len(X_train))
print("Testing images:", len(X_test))


# ==========================================
# FEATURE SCALING
# ==========================================

print("\nScaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# TRAIN SVM
# ==========================================

print("\nTraining SVM model...")
print("This may take a few minutes...")

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    probability=True,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_scaled, y_train)

print("SVM training completed!")


# ==========================================
# PREDICTION
# ==========================================

print("\nMaking predictions...")

y_pred = model.predict(X_test_scaled)


# ==========================================
# ACCURACY
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n======================================")
print("MODEL PERFORMANCE")
print("======================================")

print(f"Accuracy: {accuracy * 100:.2f}%")


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Cat", "Dog"]
    )
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# ==========================================
# SAVE CONFUSION MATRIX
# ==========================================

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Cat", "Dog"]
)

display.plot()

plt.title("Cat vs Dog - SVM Confusion Matrix")

plt.savefig(
    os.path.join(
        IMAGE_FOLDER,
        "confusion_matrix.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nConfusion matrix saved.")


# ==========================================
# SAVE MODEL
# ==========================================

model_path = os.path.join(
    MODEL_FOLDER,
    "cat_dog_svm.pkl"
)

scaler_path = os.path.join(
    MODEL_FOLDER,
    "hog_scaler.pkl"
)

joblib.dump(model, model_path)

joblib.dump(scaler, scaler_path)


print("\n======================================")
print("FILES SAVED")
print("======================================")

print("Model:", model_path)
print("Scaler:", scaler_path)

print("\nTraining completed successfully! 🎉")