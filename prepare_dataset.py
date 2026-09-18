import os
import shutil
from pathlib import Path

# -----------------------------
# SETTINGS
# -----------------------------

SOURCE = Path("kagglecatsanddogs_5340/PetImages")

CAT_SOURCE = SOURCE / "Cat"
DOG_SOURCE = SOURCE / "Dog"

CAT_DEST = Path("data/cats")
DOG_DEST = Path("data/dogs")

# Number of images we want
MAX_IMAGES = 2000


# -----------------------------
# CREATE DESTINATION FOLDERS
# -----------------------------

CAT_DEST.mkdir(parents=True, exist_ok=True)
DOG_DEST.mkdir(parents=True, exist_ok=True)


# -----------------------------
# FUNCTION TO COPY IMAGES
# -----------------------------

def copy_images(source_folder, destination_folder, limit):

    count = 0

    for file in source_folder.iterdir():

        if count >= limit:
            break

        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]:

            try:
                destination = destination_folder / file.name

                shutil.copy2(file, destination)

                count += 1

            except Exception:
                print("Skipped:", file.name)

    return count


# -----------------------------
# COPY CATS AND DOGS
# -----------------------------

print("Preparing dataset...\n")

cats = copy_images(
    CAT_SOURCE,
    CAT_DEST,
    MAX_IMAGES
)

dogs = copy_images(
    DOG_SOURCE,
    DOG_DEST,
    MAX_IMAGES
)


# -----------------------------
# RESULT
# -----------------------------

print("\n==============================")
print("DATASET PREPARATION COMPLETE")
print("==============================")

print("Cats copied:", cats)
print("Dogs copied:", dogs)

print("\nDataset structure:")
print("data/cats/")
print("data/dogs/")