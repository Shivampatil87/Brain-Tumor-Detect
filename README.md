# 🧠 NeuroScan AI | Brain Tumor Detection System

<p align="center">
  <strong>AI-powered Brain MRI Classification using Deep Learning</strong>
</p>

<p align="center">
  <a href="https://brain-tumor-detect-eef3.onrender.com/">
    <img src="https://img.shields.io/badge/Live%20Demo-Open%20App-16a34a?style=for-the-badge" alt="Live Demo"/>
  </a>
  <a href="https://github.com/Shivampatil87/Brain-Tumor-Detect">
    <img src="https://img.shields.io/badge/GitHub-Source%20Code-181717?style=for-the-badge&logo=github" alt="GitHub Repository"/>
  </a>
</p>

NeuroScan AI is a web-based brain tumor classification project built with **Python, Flask, TensorFlow/Keras, and OpenCV**. It processes uploaded brain MRI images and uses a Convolutional Neural Network (CNN) to classify them into tumor and no-tumor categories.

🚀 **Live Application:** https://brain-tumor-detect-eef3.onrender.com/

> **Medical Disclaimer:** This project is intended for educational and research purposes only. It is not a clinically validated diagnostic device. Predictions must not be used to diagnose or rule out a brain tumor or guide medical treatment.

## ✨ Features

- 🧠 **Deep Learning Classification:** CNN-based classification of brain MRI images.
- 🖼️ **MRI Image Upload:** Upload supported JPG and PNG images through the web interface.
- 🔬 **OpenCV Preprocessing:** Brain contour detection and cropping to reduce distracting background regions.
- 📊 **Prediction Results:** Displays the predicted category and associated confidence information.
- 🌐 **Flask Web Application:** Interactive browser-based interface.
- 📱 **Responsive UI:** Designed for a convenient user experience.
- 🐳 **Docker Support:** Containerized deployment configuration.
- ☁️ **Cloud Deployment:** Hosted online using Render.

## 🌐 Live Demo

**Try the application:** [NeuroScan AI - Brain Tumor Detection](https://brain-tumor-detect-eef3.onrender.com/)

1. Open the live website.
2. Navigate to the MRI detection page.
3. Upload a supported brain MRI image.
4. Submit the image for processing.
5. Review the application's predicted category and displayed confidence.

The results are model predictions, not medical diagnoses. Availability and response time may vary depending on hosting resources.

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Backend | Flask |
| Deep Learning | TensorFlow, Keras |
| Image Processing | OpenCV, NumPy, imutils |
| Frontend | HTML5, CSS3, JavaScript |
| Model Format | Keras HDF5 (`.h5`) |
| Containerization | Docker |
| Hosting | Render |
| Version Control | Git, GitHub |

## 📁 Project Structure

```text
Brain-Tumor-Detect/
├── app.py
├── train_model.py
├── requirements.txt
├── Dockerfile
├── README.md
├── .gitignore
│
├── model/
│   └── brain_tumor_model.h5
│
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── main.js
│   └── images/
│       ├── training_history.png
│       └── confusion_matrix.png
│
├── templates/
│   ├── base.html
│   ├── home.html
│   ├── detect.html
│   ├── result.html
│   ├── about.html
│   └── contact.html
│
├── uploads/
└── dataset/
    ├── yes/
    └── no/
```

*The structure above describes the intended project layout. Some folders, generated graphs, or the dataset may not be present in a fresh clone. Keep private MRI uploads and sensitive patient data out of the public repository.*

## ⚙️ Run the Project Locally

### Prerequisites

- Python 3.11, compatible with the supplied Docker base image and installed dependencies
- pip
- Git
- Optional: Docker

### 1. Clone the repository

```bash
git clone https://github.com/Shivampatil87/Brain-Tumor-Detect.git
cd Brain-Tumor-Detect
```

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS/Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If installation fails, check the Python version and the compatibility of the TensorFlow and OpenCV packages specified in `requirements.txt`.

### 4. Prepare the dataset

Download the dataset from Kaggle:

[Brain Tumor Detection Dataset - Kaggle](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection)

Arrange the dataset as expected by the training script:

```text
dataset/
├── yes/
│   ├── tumor_image_1.jpg
│   └── tumor_image_2.jpg
└── no/
    ├── normal_image_1.jpg
    └── normal_image_2.jpg
```

The folder names and labels must match the implementation in `train_model.py`.

### 5. Train the model

If the trained model is not already available, run:

```bash
python train_model.py
```

The script should save the trained model and evaluation plots if those output operations are implemented.

Expected output paths:

```text
model/brain_tumor_model.h5
static/images/training_history.png
static/images/confusion_matrix.png
```

Training time depends on the hardware, dependency versions, dataset and configuration.

**Note:** Retraining may produce different results. The reported metrics below should be verified against the original notebook or evaluation logs.

### 6. Start the application

```bash
python app.py
```

Open your browser at:

http://127.0.0.1:5000/

Upload an MRI image and test the application's prediction workflow.

## 🧠 CNN Model Architecture

The model uses an input image size of `240 × 240 × 3` and a binary output.

```text
Input Image (240 × 240 × 3)
          |
      Zero Padding
          |
   Conv2D (32 filters)
       Kernel 7 × 7
          |
 Batch Normalization
          |
        ReLU
          |
   Max Pooling (4 × 4)
          |
   Max Pooling (4 × 4)
          |
        Flatten
          |
     Dense (1 neuron)
          |
   Sigmoid Activation
          |
 Binary Classification
```

The reported model has approximately **11,137 parameters**. Verify the exact layer configuration and parameter count using `model.summary()`.

### Image preprocessing

The preprocessing pipeline uses OpenCV-based brain contour cropping.

1. Convert the MRI image to grayscale.
2. Apply Gaussian blur to reduce noise.
3. Apply binary thresholding.
4. Perform erosion and dilation.
5. Detect external contours and select the largest contour.
6. Crop the image using the contour's extreme points.
7. Resize and normalize the image to the model's expected input format.

Training and inference should use the same preprocessing, colour-channel order, dimensions and normalization.

## 📊 Reported Model Performance

The following metrics were reported by the training notebook. They should not be interpreted as independently verified clinical performance.

| Dataset Split | Samples | Accuracy | Loss |
|---|---:|---:|---:|
| Training | 2,700 | 98.59% | 0.0548 |
| Validation | 150 | 94.67% | 0.1620 |
| Test | 150 | 95.33% | 0.1430 |
| **Total** | **3,000** | — | — |

**Evaluation note:** Accuracy alone does not establish medical reliability. Further evaluation should include sensitivity, specificity, precision, recall, confusion-matrix analysis and testing on independent data. Dataset splitting should also prevent patient-level data leakage.

## 🐳 Docker Deployment

Build the image from the project root:

```bash
docker build -t neuroscan-ai .
```

Run the container:

```bash
docker run --rm -p 5000:5000 neuroscan-ai
```

Open:

http://127.0.0.1:5000/

The Docker image must contain the required dependencies and make the trained model available at the path expected by `app.py`. Use a production WSGI server and appropriate security settings for a deployed application.

## ☁️ Deployment on Render

This project is deployed on Render.

**Live URL:** https://brain-tumor-detect-eef3.onrender.com/

To deploy your own instance:

1. Push the project to a GitHub repository.
2. Connect the repository to Render.
3. Configure the service to build using the repository's `Dockerfile`.
4. Ensure that the model file and Python dependencies are available.
5. Review the build and runtime logs.
6. Test image upload and prediction after deployment.

For deployment configuration and service options, refer to the [Render documentation](https://render.com/docs).

## 🔒 Security and Privacy

- Validate uploaded image formats and file sizes.
- Use safe server-generated filenames.
- Remove temporary uploaded images when no longer required.
- Never commit identifiable patient scans, private data, API keys or credentials.
- Keep environment-specific settings outside source control.
- Use appropriate production server settings and access controls.

## ⚠️ Medical Disclaimer

NeuroScan AI is a student project intended for learning and research in medical image classification.

The model may produce incorrect predictions. Displayed confidence values are not necessarily calibrated probabilities and must not be treated as the likelihood that a patient has a tumor.

**Do not use this application for clinical diagnosis, treatment decisions, or to rule out a brain tumor.** Consult qualified healthcare professionals for medical assessment.

## 🚀 Future Improvements

- Evaluate sensitivity, specificity, precision, recall and ROC-AUC.
- Improve handling of invalid, corrupted or unsupported images.
- Add automated tests for preprocessing and prediction.
- Evaluate the model on independent datasets.
- Improve model robustness across scanners and imaging protocols.
- Strengthen upload validation, privacy controls and deployment monitoring.

## 👨‍💻 Author

**Shivam Mahalle**

- GitHub: [@Shivampatil87](https://github.com/Shivampatil87)
- Repository: [Brain-Tumor-Detect](https://github.com/Shivampatil87/Brain-Tumor-Detect)
- Live Demo: [NeuroScan AI](https://brain-tumor-detect-eef3.onrender.com/)

---

⭐ If you find this project useful for learning about deep learning and computer vision, consider starring the repository.

**Built for educational research in AI-assisted medical image classification.**
