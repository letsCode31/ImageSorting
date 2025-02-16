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

# 🖼️ Face Sorting Tool - HOG & CNN-Based

This project automates **image sorting based on face recognition**. It scans a folder of images, detects **your face**, and moves matched images to a separate folder.

It provides **two approaches**:

1. **HOG-Only (**``**)** → Faster but may miss difficult cases.
2. **HOG + CNN (**``**)** → More accurate, runs CNN on unmatched images.

---

## 🚀 Features

✅ **Face Detection & Recognition** using `face_recognition` (HOG/CNN).\
✅ **Preprocessing for Better Accuracy** (Contrast Enhancement, Image Sharpening).\
✅ **Two-Step Processing (HOG + CNN)** to reduce false negatives.\
✅ **Automatic Folder Management** (Sorted & Unmatched Images).

---

## 📂 Folder Structure I used in VS Code 

```
face-sorting-tool/
│── reference_images/      # Your sample images for face matching
│── source_images/         # Images to be sorted
│── sorted_images/         # Matched images are moved here
│── unmatched_images/      # Unmatched images are moved here
│── sort_images_hog_2.py   # HOG-based sorting script
│── sort_images_hog_modified.py  # HOG + CNN sorting script
│── requirements.txt       # Python dependencies
```

---

## 🔧 Installation & Setup

### **1️⃣ Install Dependencies**

Make sure you have **Python 3.8+** installed, then install the required packages:

```sh
pip install -r requirements.txt
```

### **2️⃣ Add Reference Images**

Place **15-20 clear images of your face** in ``.\
💡 These should cover **different angles, lighting, and expressions** for better accuracy.

### **3️⃣ Place Images to Sort**

Copy **all images you want to sort** into `source_folder`.

---

## ▶️ Running the Sorting Script

### **Option 1: Using HOG Only (Faster, Less Accurate)**

This approach uses only the **HOG model** to detect and sort images.

Run the following command:

```sh
python sort_images_hog_2.py
```

✅ This will:

- **Move matched images** → `sorted_images/`
- **Move unmatched images** → `unmatched_images/`

---

### **Option 2: Using HOG + CNN (More Accurate)**

This approach first runs **HOG**, then **uses CNN to recheck unmatched images**.

Run:

```sh
python sort_images_hog_modified.py
```

✅ This will:

- **First pass:** Run **HOG**, moving images to `sorted_images/` or `unmatched_images/`.
- **Second pass:** CNN **reprocesses images in **`unmatched_images` and moves new matches to `sorted_images/`.

💡 This **reduces false negatives**, ensuring better accuracy!

---

## 🔍 Results Explained

| Folder              | Description                                     |
| ------------------- | ----------------------------------------------- |
| `sorted_images/`    | Contains images where your face was detected. ✅ |
| `unmatched_images/` | Contains images where no match was found. ❌     |

💡 **In the HOG + CNN approach**, images in `unmatched_images/` get rechecked using CNN. If a match is found, they move to `sorted_images/`. Otherwise, they remain unmatched.

---

## ⚡ Troubleshooting & Fine-Tuning

### **1️⃣ No Faces Detected?**

- Ensure **clear, front-facing images** in `reference_images/`.
- Add **more diverse images** (lighting, angles, expressions).

### **2️⃣ Too Many False Negatives (Missed Matches)?**

- Increase threshold in script (`tolerance=0.48` → `0.50`).
- Use the **HOG + CNN version** (`sort_images_hog_modified.py`).

### **3️⃣ False Positives (Wrong Matches)?**

- Lower threshold (`tolerance=0.48` → `0.45`).
- Check `reference_images/` for **misidentified faces**.

---

## 📝 Notes

- **Processing time:**
  - `HOG-only` is **faster**.
  - `HOG + CNN` is **slower** but **more accurate**.
  - ** Disclaimer** The time frames may vary depending your system power. Since I was using a Intel based 2012 macOS, my processing speed was very slow for CNN. however HOG model worked fine.  
- **Ideal dataset size:** At least **10-15 reference images**.

---




