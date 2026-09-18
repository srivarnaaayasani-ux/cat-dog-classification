import cv2
import joblib
import sys

from skimage.feature import hog


# ==========================================
# SETTINGS
# ==========================================

IMAGE_SIZE = (128, 128)

MODEL_PATH = "models/cat_dog_svm.pkl"
SCALER_PATH = "models/hog_scaler.pkl"


# ==========================================
# CHECK IMAGE PATH
# ==========================================

if len(sys.argv) < 2:
    print("Please provide an image path.")
    print("Example:")
    print("python predict.py test_images/cat.jpg")
    sys.exit()


image_path = sys.argv[1]


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print("Model loaded successfully!")


# ==========================================
# LOAD IMAGE
# ==========================================

image = cv2.imread(image_path)

if image is None:
    print("Error: Could not open the image.")
    print("Check the image path.")
    sys.exit()


# ==========================================
# RESIZE IMAGE
# ==========================================

image = cv2.resize(image, IMAGE_SIZE)


# ==========================================
# CONVERT TO GRAYSCALE
# ==========================================

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)


# ==========================================
# EXTRACT HOG FEATURES
# ==========================================

features = hog(
    gray,
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys"
)


# ==========================================
# RESHAPE FEATURES
# ==========================================

features = features.reshape(1, -1)


# ==========================================
# SCALE FEATURES
# ==========================================

features_scaled = scaler.transform(features)


# ==========================================
# MAKE PREDICTION
# ==========================================

prediction = model.predict(features_scaled)[0]

probabilities = model.predict_proba(features_scaled)[0]

confidence = max(probabilities) * 100


# ==========================================
# CONVERT LABEL TO NAME
# ==========================================

if prediction == 0:
    result = "CAT 🐱"
else:
    result = "DOG 🐶"


# ==========================================
# DISPLAY RESULT
# ==========================================

print("\n================================")
print("       CAT vs DOG CLASSIFIER")
print("================================")

print("Prediction :", result)
print(f"Confidence : {confidence:.2f}%")

print("================================")