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

## Scripts

The `scripts/` folder contains modular components for the entire training and evaluation pipeline. Here is an overview of each script:

- **data.py**  
  Contains custom dataset classes and transformation pipelines for chest X-ray segmentation.  
  **Key Classes:**  
  - `ChestXRayDataset`: Loads images and their corresponding masks from specified directories.  
  - `JointTransformWrapper`: Applies transformations (resizing, normalization, random horizontal flips, etc.) simultaneously to both images and masks.

- **model.py**  
  Defines the segmentation model architecture.  
  **Components:**  
  - `SimpleDecoder`: A UNet-like decoder without attention blocks using skip connections and transposed convolutions.  
  - `SwinTransformerSegModel`: Integrates a pretrained Swin Transformer encoder with the custom decoder for segmentation.

- **losses.py**  
  Contains loss functions and metric calculations for training and evaluation.  
  **Functions:**  
  - `dice_loss` and `ComboLoss`: Compute the weighted combination of BCE and Dice loss.  
  - Metrics such as `dice_coefficient`, `iou_coefficient`, and functions for confusion matrix computation and additional performance metrics.

- **utils.py**  
  Provides utility functions to support data loading and time formatting.  
  **Functions:**  
  - `create_dataloaders`: Creates dataloaders for training, validation, and test sets.  
  - `format_time`: Converts seconds into an hh:mm:ss formatted string.

- **train.py**  
  Implements the complete training loop for the segmentation model.  
  **Features:**  
  - Loads and splits the dataset into training, validation, and test sets.  
  - Trains the model with early stopping and learning rate scheduling.  
  - Saves the best model weights and plots training curves.  
  **Usage Example:**  
  ```
  python scripts/train.py --dataset_path <path_to_dataset> --epochs 20 --batch_size 4 --output_dir outputs
  ```

- **evaluate.py**  
  Evaluates a saved model on the test set and prints out performance metrics (loss, Dice, IoU, accuracy, etc.).  
  **Usage Example:**  
  ```
  python scripts/evaluate.py --dataset_path <path_to_dataset> --model_path <path_to_model_weights> --batch_size 4
  ```

- **visualize.py**  
  Provides functions to visualize model predictions alongside ground truth and overlayed masks.  
  **Usage Example:**  
  ```
  python scripts/visualize.py --dataset_path <path_to_dataset> --batch_size 4
  ```

These scripts provide a modular and scalable framework to train, evaluate, and visualize your chest X-ray segmentation model.

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
