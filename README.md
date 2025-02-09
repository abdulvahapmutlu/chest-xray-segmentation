# Chest X-Ray Segmentation with Swin Transformer

An advanced, production-ready segmentation project that leverages a Swin Transformer-based model to perform chest X-ray segmentation. This repository includes model training and evaluation code, a Streamlit web application for inference and visualization, containerization with Docker, deployment manifests for Kubernetes, and a CI/CD pipeline using GitHub Actions.

---

## Overview

This project implements a chest X-ray segmentation model using a Swin Transformer as the encoder and a custom UNet-like decoder without attention blocks. The segmentation network was trained using a custom dataset and achieves high performance (e.g., Dice score of ~0.96 and IoU of ~0.92 on the test set). A Streamlit-based web interface enables users to upload images, run inference, and visualize the original image, predicted segmentation mask, and an overlay of the mask on the image. This repository also includes Docker and Kubernetes configurations for production deployment and a GitHub Actions workflow for CI/CD.

---

## Features

- **Advanced Segmentation Model:**  
  Uses a Swin Transformer backbone with a custom decoder for efficient segmentation of chest X-rays.
  
- **Comprehensive Training & Evaluation:**  
  Includes early stopping, dynamic learning rate scheduling, and extended performance metrics (accuracy, precision, recall, specificity, F1-score).

- **Interactive Web Application:**  
  A Streamlit app for uploading images, adjusting segmentation parameters, and visualizing results with overlays.

- **Containerization & Orchestration:**  
  Dockerfile provided for building container images and Kubernetes manifests for scalable deployment.

- **CI/CD Pipeline:**  
  Automated build, lint, and push of Docker images via GitHub Actions.

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) Docker and Kubernetes CLI tools if deploying in a containerized environment

### Local Setup

1. **Clone the Repository:**

   ```
   git clone https://github.com/abdulvahapmutlu/chest-xray-segmentation.git
   cd chest-xray-segmentation
   ```

2. **Create and Activate a Virtual Environment (Recommended):**

   ```
   python -m venv venv
   source venv/bin/activate      # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies:**

   ```
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Usage

### Running Locally

To run the Streamlit web app locally:

```
streamlit run app.py
```

- **Upload an Image:** Use the file uploader in the app to upload a chest X-ray image.
- **Adjust Settings:** Use the sidebar to adjust the segmentation threshold and overlay alpha value.
- **View Results:** After running inference, view the original image, segmentation mask, and overlay.
- **Download Overlay:** Click the download button to save the overlay image.

### Docker Deployment

1. **Build the Docker Image:**

   ```
   docker build -t <your-docker-username>/streamlit-app:latest -f docker/Dockerfile .
   ```

2. **Run the Docker Container:**

   ```
   docker run -p 8501:8501 <your-docker-username>/streamlit-app:latest
   ```

3. **Access the App:**  
   Open your web browser and navigate to `http://localhost:8501`.

### Kubernetes Deployment

1. **Update Manifests:**  
   In `kubernetes/streamlit-deployment.yaml`, replace `<YOUR_DOCKER_IMAGE>` with your actual Docker image (e.g., `your-docker-username/streamlit-app`).

2. **Deploy to Kubernetes:**

   ```
   kubectl apply -f kubernetes/streamlit-deployment.yaml
   kubectl apply -f kubernetes/streamlit-service.yaml
   ```

3. **Access the App:**  
   Depending on your Kubernetes setup, the service will expose the app via a LoadBalancer or NodePort.

---

## CI/CD Pipeline

This project contains CI/CD Demo:

- **Workflow File:** `ci-cd-demo.txt`
- **Actions Performed:**
  - Checkout the repository.
  - Set up Python and install dependencies.
  - Lint the code using flake8.
  - Build the Docker image.
  - Log in to Docker Hub using secrets (`DOCKER_USERNAME` and `DOCKER_PASSWORD`).
  - Push the Docker image to Docker Hub.

---

## Training Results

The training results (saved in `outputs`) provide detailed evaluation metrics.

=== Test Results ===
Loss: 0.0548
Dice: 0.9591 | IoU: 0.9227
Accuracy: 0.9803
Precision: 0.9657
Recall: 0.9567
Specificity: 0.9883
F1: 0.9612

## Notebook

A Jupyter notebook file is in `notebook` folder if you'd like to train the model. 


## License

This project is licensed under the [MIT License](LICENSE).
