# 🌿 Plants Disease Prediction

A deep learning-based image classification project designed to detect and identify diseases in plant leaves. This project helps farmers, researchers, and agriculturists diagnose crop diseases early by analyzing leaf images through a trained model.

---

## 🧠 Project Overview

This project uses convolutional neural networks (CNNs) and computer vision techniques to classify plant diseases from images. The model is trained on a dataset of healthy and diseased leaf images across various plant species. The application is intended to help automate plant disease detection using image input, potentially from mobile or web platforms.

---

## 📁 Repository Structure

```bash
Plants_Disease_Prediction/
│
├── PlantVillage/                 # Dataset directory (can be downloaded separately)
│
├── Disease_Detection_Model/     # Trained model (.h5) and training scripts
│   ├── model.h5                 # Trained Keras/TensorFlow model
│   └── train_model.ipynb        # Notebook used for training the model
│
├── Predict/                     # Prediction scripts and input
│   ├── predict.py               # Script to predict disease from an image
│   ├── test_leaf.jpg            # Example image for prediction
│
├── gui/                         # GUI built with Tkinter
│   └── gui.py                   # Python file to run the desktop app
│
├── requirements.txt             # Python dependencies
├── README.md                    # Project overview and instructions
└── LICENSE                      # License file
```

---

## 📸 Dataset

The dataset used in this project is the **PlantVillage** dataset, which contains over 50,000 images of healthy and diseased crop leaves categorized into:

* Apple
* Potato
* Tomato
* Grape
* Corn
* ...and more

Each class includes both healthy and diseased samples for robust model training.

**Dataset Source:** [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)

---

## 🚀 Features

* Detects plant diseases from leaf images
* Trained using CNN with TensorFlow/Keras
* Easy-to-use GUI for desktop users (Tkinter)
* CLI-based prediction available
* Customizable model training and evaluation

---

## 🛠️ Installation

### 🔧 Prerequisites

* Python 3.7+
* pip

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🧪 Usage

### 1. Train the Model (Optional)

If you want to retrain the model on your own dataset:

```bash
cd Disease_Detection_Model
jupyter notebook train_model.ipynb
```

> Note: You need to download the PlantVillage dataset and configure paths.

### 2. Predict Using CLI

```bash
cd Predict
python predict.py --image test_leaf.jpg
```
**3. Run GUI Application**

```bash
cd gui
python gui.py
```

The GUI allows users to upload an image and get predictions interactively.

🎯 Model Architecture

* Model Type: CNN (Convolutional Neural Network)
* Framework: TensorFlow / Keras
* Optimizer: Adam
* Loss Function: Categorical Crossentropy
* Metrics: Accuracy

📊 Evaluation

The model achieves high accuracy on the validation set and performs well on unseen test images. Evaluation metrics like accuracy, confusion matrix, and classification reports are included in the training notebook.

🧑‍💻 Author

Salman Rasheed
GitHub: [@SalmanRasheed568](https://github.com/SalmanRasheed568)



📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

 🙌 Acknowledgements

* [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
* TensorFlow and Keras documentation
* OpenCV and Pillow libraries


