import os
import requests
import zipfile
import shutil

URL = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"

ZIP_FILE = "cats_and_dogs_filtered.zip"
EXTRACT_FOLDER = "cats_and_dogs_filtered"

print("Downloading Cats vs Dogs dataset...")

headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(URL, headers=headers)

if response.status_code != 200:
    print("Download failed!")
    print("Status code:", response.status_code)
    exit()

with open(ZIP_FILE, "wb") as file:
    file.write(response.content)

print("Download completed!")

print("Extracting dataset...")

with zipfile.ZipFile(ZIP_FILE, "r") as zip_ref:
    zip_ref.extractall(".")

print("Extraction completed!")

# Remove old folders
if os.path.exists("data/cats"):
    shutil.rmtree("data/cats")

if os.path.exists("data/dogs"):
    shutil.rmtree("data/dogs")

# Create folders
os.makedirs("data/cats", exist_ok=True)
os.makedirs("data/dogs", exist_ok=True)

# Source folders
source_cats = os.path.join(
    EXTRACT_FOLDER,
    "train",
    "cats"
)

source_dogs = os.path.join(
    EXTRACT_FOLDER,
    "train",
    "dogs"
)

# Copy cats
for file in os.listdir(source_cats):
    shutil.copy2(
        os.path.join(source_cats, file),
        os.path.join("data/cats", file)
    )

# Copy dogs
for file in os.listdir(source_dogs):
    shutil.copy2(
        os.path.join(source_dogs, file),
        os.path.join("data/dogs", file)
    )

print("\n==============================")
print("DATASET READY!")
print("==============================")

print("Cats:", len(os.listdir("data/cats")))
print("Dogs:", len(os.listdir("data/dogs")))

print("\nDataset location:")
print("data/cats/")
print("data/dogs/")