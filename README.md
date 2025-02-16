# 🖼️ Image Sorting Tool - HOG & CNN Based

This project automates image sorting by identifying specific faces from a folder of images.  
It uses **Histogram of Oriented Gradients (HOG)** and **Convolutional Neural Networks (CNN)** for face detection and recognition.

---

## 🚀 Features
✅ **Two Versions of Sorting**  
- **`sort_images_hog_2.py`** → Uses only **HOG-based face recognition**.  
- **`sort_images_hog_modified.py`** → First runs **HOG**, then uses **CNN** for rechecking unmatched images.  

✅ **Preprocessing Enhancements**  
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)** → Enhances low-light images.  
- **Image Sharpening** → Improves detection accuracy.  

✅ **Automatic Folder Management**  
- `sorted_images/` → Images that match reference faces.  
- `unmatched_images/` → Images that don’t match reference faces.  

---

## 📂 Folder Structure
