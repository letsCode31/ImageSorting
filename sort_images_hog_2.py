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
    image = Image.open(image_path).convert("RGB")  # Ensure it's RGB
    enhancer = ImageEnhance.Sharpness(image)
    enhanced_image = enhancer.enhance(factor)
    
    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(enhanced_image, dtype=np.uint8), cv2.COLOR_RGB2BGR)

# Function to denoise and sharpen images using OpenCV
def enhance_image_quality(image_path):
    img = cv2.imread(image_path)

    if img is None:
        print(f"⚠️ Could not read {image_path}")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur (reduces noise)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Use Unsharp Masking to enhance sharpness
    sharpened = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

    # Convert back to RGB and ensure uint8 format
    sharpened_rgb = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
    return np.array(sharpened_rgb, dtype=np.uint8)  # Ensure uint8 format

# Load reference images and compute face encodings
reference_encodings = []

print("\n⏳ Loading and enhancing reference images...")
for filename in os.listdir(REFERENCE_FOLDER):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        ref_path = os.path.join(REFERENCE_FOLDER, filename)

        # Enhance sharpness of reference images
        enhanced_image = enhance_sharpness(ref_path, factor=2.0)

        # Apply CLAHE for better contrast
        enhanced_image = apply_clahe(enhanced_image)

        # Convert to RGB for face recognition
        enhanced_image_rgb = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)

        # Detect faces using HOG-based detection
        face_locations = face_recognition.face_locations(enhanced_image_rgb, model="hog")
        encodings = face_recognition.face_encodings(enhanced_image_rgb, face_locations)

        if encodings:
            reference_encodings.append(encodings[0])
            print(f"✅ Loaded and enhanced reference image: {filename}")
        else:
            print(f"⚠️ No face found in reference image: {filename}")

if not reference_encodings:
    print("\n❌ ERROR: No valid face encodings found in the reference images!")
    exit()

print(f"✅ {len(reference_encodings)} reference faces loaded!\n")

# Process each image in the source folder
print("🔍 Scanning and enhancing images in source folder...\n")

for filename in os.listdir(SOURCE_FOLDER):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Ensure it's an image
        src_path = os.path.join(SOURCE_FOLDER, filename)

        try:
            # Load the image and apply CLAHE
            image = cv2.imread(src_path)
            if image is None:
                print(f"⚠️ Could not read {filename}, skipping.")
                continue
            
            # Enhance source image before face recognition
            enhanced_image1 = enhance_image_quality(src_path)
            if enhanced_image1 is None:
                continue  # Skip processing if enhancement fails
            
            enhanced_image = apply_clahe(enhanced_image1)

            # Convert to RGB for face recognition
            enhanced_image_rgb = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)

        except Exception as e:
            print(f"⚠️ Could not process {filename}: {e}")
            continue
            
        # Detect face locations before encoding
        face_locations = face_recognition.face_locations(enhanced_image_rgb, model="hog")
        face_encodings = face_recognition.face_encodings(enhanced_image_rgb, face_locations)

        if not face_encodings:
            print(f"🚫 No faces detected in {filename}. Moving to unmatched folder.")
            shutil.move(src_path, os.path.join(UNMATCHED_FOLDER, filename))
            continue

        # Compare each detected face with reference encodings
        match_found = False
        threshold = 0.48  # Tighter threshold to reduce false positives

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(reference_encodings, face_encoding, tolerance=threshold)
            if any(matches):  # At least one match
                match_found = True
                break  # Stop checking once a match is found

        if match_found:
            shutil.move(src_path, os.path.join(DESTINATION_FOLDER, filename))
            print(f"✅ Match found! Moved: {filename}")
        else:
            shutil.move(src_path, os.path.join(UNMATCHED_FOLDER, filename))
            print(f"❌ No match for: {filename}. Moved to unmatched folder.")

print("\n🎯 Sorting complete! Check 'sorted_images' and 'unmatched_images' folders.\n")
