# HematoVision-Advanced-Blood-Cell-Classification-Using-Transfer-Learning
AICTE Internship



**Team ID:** LTVIP2026TMIDS65921  
**Team Size:** 4

**Team Leader:** Anumala Vishnuvardhan  
**Team Members:**
- Bellamkonda Harshini
- Chinthala Sandeep
- Garikapati Likhitha

## Project Overview
Hematovision is a deep learning project designed to classify blood cell images into four distinct categories: **Eosinophil, Lymphocyte, Monocyte, and Neutrophil**. The project utilizes **Transfer Learning** with the **MobileNetV2** architecture to achieve high accuracy. It includes a training script to build the model and a Flask-based web application for user-friendly interaction.

## Technologies Used
- **Python**
- **TensorFlow / Keras** (Deep Learning)
- **MobileNetV2** (Transfer Learning Model)
- **Flask** (Web Framework)
- **HTML/CSS** (Frontend)

## Dataset
This project uses the [Blood Cell Images Dataset](https://www.kaggle.com/datasets/paultimothymooney/blood-cells/data) from Kaggle.

### Dataset Setup
1. Download the dataset from Kaggle.
2. Extract the files.
3. Create a folder named `dataset` inside the `Project files` directory.
4. Organize the folders as follows:
   ```text
   Project files/
   ├── dataset/
   │   ├── TRAIN/
   │   │   ├── EOSINOPHIL/
   │   │   ├── ...
   │   └── TEST/
   │       ├── EOSINOPHIL/
   │       ├── ...
   ```

## Installation
1. Navigate to the project directory.
2. Install the required dependencies:
   ```bash
   pip install -r "Project files/requirements.txt"
   ```

## Usage

### 1. Training the Model
To train the model from scratch, run the `app.py` script. This will generate the `blood_cell_classifier_mobilenetv2.h5` file.

```bash
cd "Project files"
python app.py
```

### 2. Running the Web Application
Once the model is trained, you can start the web interface to classify new images.

```bash
cd "Project files"
python web_app.py
```
Open your browser and go to `http://127.0.0.1:5000`.
