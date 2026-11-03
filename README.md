# Potato vs. Onion Image Classifier 🥔🧅

A lightweight, full-stack deep learning web application that classifies produce images (Potato vs. Onion) using PyTorch, FastAPI, and an interactive HTML/JavaScript frontend.

---

## 🚀 Project Overview
This project streamlines image classification into two distinct classes (`potatoes` and `onions`). It includes:
- **PyTorch Training Pipeline (`train_model.py`)**: Supports custom CNNs and transfer learning with MobileNetV2.
- **FastAPI Backend (`app.py`)**: Serves the trained model via REST endpoints with confidence thresholding and CORS enabled.
- **Web Frontend**: Simple, clean user interface to upload images and view classification results in real time.

---

## 🛠️ Prerequisites & Installation

Make sure you have **Python 3.9+** installed on your system.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/potato_vs_onion.git
   cd potato_vs_onion
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📂 Dataset Structure
Organize your training data inside a `dataset/` folder in the root directory following this layout:

```text
potato_vs_onion/
│
└── dataset/
    ├── onions/
    │   ├── image1.jpg
    │   └── image2.jpg
    └── potatoes/
        ├── image1.jpg
        └── image2.jpg
```

---

## 🏋️‍♂️ Training the Model
To train the model from scratch or fine-tune MobileNetV2 on your dataset, run:

```bash
python train_model.py --model mobilenet_v2 --epochs 25 --batch-size 32
```
*This will automatically generate the training outputs (including `best_model.pth` and `classes.json`) inside an `outputs/` folder.*

---

## 🔌 Running the FastAPI Backend
Once your model is trained, start the FastAPI backend server using Uvicorn:

```bash
uvicorn app:app --reload
```
The server will run locally at `http://127.0.0.1:8000`.

---

## 🌐 Testing the Application
1. Open the frontend HTML file (`frontend/potato_vs_onion_classifier_frontend.html`) directly in your web browser.
2. (Optional) If accessing via a public tunnel, run **ngrok**:
   ```bash
   ngrok http 8000
   ```
3. Upload an image of a potato or an onion to test real-time classification!