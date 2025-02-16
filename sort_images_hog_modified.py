import os
import shutil
import face_recognition
import cv2
from PIL import Image, ImageEnhance
import numpy as np

# Define folder paths
SOURCE_FOLDER = "source_images"
DESTINATION_FOLDER = "sorted_images"
UNMATCHED_FOLDER = "unmatched_images"
REFERENCE_FOLDER = "reference_images"

os.makedirs(DESTINATION_FOLDER, exist_ok=True)
os.makedirs(UNMATCHED_FOLDER, exist_ok=True)

# Function to apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
def apply_clahe(image):
    """
    Enhances contrast using CLAHE to improve face detection in low-light or poor-contrast images.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equalized = clahe.apply(gray)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)

# Function to enhance sharpness of an image
def enhance_sharpness(image_path, factor=2.0):
    image = Image.open(image_path).convert("RGB")
    enhancer = ImageEnhance.Sharpness(image)
    enhanced_image = enhancer.enhance(factor)
    return cv2.cvtColor(np.array(enhanced_image, dtype=np.uint8), cv2.COLOR_RGB2BGR)

# Load reference images for HOG and CNN
reference_encodings = []

print("\n⏳ Loading and enhancing reference images...")
for filename in os.listdir(REFERENCE_FOLDER):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        ref_path = os.path.join(REFERENCE_FOLDER, filename)

        # Process for HOG & CNN
        enhanced_image = enhance_sharpness(ref_path, factor=2.0)
        enhanced_image = apply_clahe(enhanced_image)
        enhanced_image_rgb = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)

        # Detect faces using HOG (for reference encodings)
        face_locations = face_recognition.face_locations(enhanced_image_rgb, model="hog")
        encodings = face_recognition.face_encodings(enhanced_image_rgb, face_locations)

        if encodings:
            reference_encodings.append(encodings[0])
            print(f"✅ Loaded reference image: {filename}")
        else:
            print(f"⚠️ No face found in reference image: {filename}")

if not reference_encodings:
    print("\n❌ ERROR: No valid face encodings found in reference images!")
    exit()

print(f"✅ {len(reference_encodings)} reference faces loaded!\n")

# Step 1: HOG-based sorting
print("🔍 Running HOG model for initial sorting...\n")

for filename in os.listdir(SOURCE_FOLDER):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        src_path = os.path.join(SOURCE_FOLDER, filename)

        try:
            image = cv2.imread(src_path)
            if image is None:
                print(f"⚠️ Could not read {filename}, skipping.")
                continue

            enhanced_image = apply_clahe(image)
            enhanced_image_rgb = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)

        except Exception as e:
            print(f"⚠️ Could not process {filename}: {e}")
            continue

        # Detect face locations
        face_locations = face_recognition.face_locations(enhanced_image_rgb, model="hog")
        face_encodings = face_recognition.face_encodings(enhanced_image_rgb, face_locations)

        if not face_encodings:
            print(f"🚫 No faces detected in {filename}. Moving to unmatched folder.")
            shutil.move(src_path, os.path.join(UNMATCHED_FOLDER, filename))
            continue

        match_found = False
        threshold = 0.48  

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(reference_encodings, face_encoding, tolerance=threshold)
            if any(matches):  
                match_found = True
                break  

        if match_found:
            shutil.move(src_path, os.path.join(DESTINATION_FOLDER, filename))
            print(f"✅ Match found! Moved: {filename}")
        else:
            shutil.move(src_path, os.path.join(UNMATCHED_FOLDER, filename))
            print(f"❌ No match for: {filename}. Moved to unmatched folder.")

print("\n✔ HOG sorting complete. Now running CNN on unmatched images...\n")

# Step 2: CNN-based reprocessing
def process_unmatched_with_cnn():
    print("\n🔍 Re-checking unmatched images using CNN...\n")

    for filename in os.listdir(UNMATCHED_FOLDER):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            src_path = os.path.join(UNMATCHED_FOLDER, filename)
            
            try:
                image = cv2.imread(src_path)
                if image is None:
                    print(f"⚠️ Could not read {filename}, skipping.")
                    continue

                enhanced_image = apply_clahe(image)
                enhanced_image_rgb = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)

            except Exception as e:
                print(f"⚠️ Could not process {filename}: {e}")
                continue

            # Detect faces using CNN
            face_locations = face_recognition.face_locations(enhanced_image_rgb, model="cnn")
            face_encodings = face_recognition.face_encodings(enhanced_image_rgb, face_locations)

            if not face_encodings:
                print(f"🚫 No faces detected in {filename}.")
                continue

            match_found = False
            threshold = 0.48  # Adjusted to reduce false positives

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(reference_encodings, face_encoding, tolerance=threshold)
                if any(matches):  # At least one match
                    match_found = True
                    break

            if match_found:
                shutil.move(src_path, os.path.join(DESTINATION_FOLDER, filename))
                print(f"✅ CNN Match Found! Moved: {filename}")
            else:
                print(f"❌ No match for {filename} with CNN.")

# Run CNN processing
process_unmatched_with_cnn()

print("\n🎯 Final sorting complete! Check 'sorted_images' and 'unmatched_images' folders.\n")
