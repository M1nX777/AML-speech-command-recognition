# Robust Small-Vocabulary Speech Command Recognition: Comparing MFCC-MLP and Mel-Spectrogram-CNN Pipelines
A small-vocabulary speech command recognition project comparing an MFCC + MLP baseline with a mel-spectrogram + CNN pipeline.

## Description

This project focuses on small-vocabulary speech command recognition, also known as keyword spotting. 
The goal is to classify short spoken audio clips into one of eight command labels: yes, no, up, down, left, right, stop, or go.

The project uses the Google Speech Commands dataset filtered to these eight commands. It compares two supervised multiclass classification pipelines:
an MFCC-based multilayer perceptron baseline and a mel-spectrogram-based convolutional neural network. 
The project also evaluates how robust both pipelines are when background noise is added to the audio.

## Getting Started

### Dependencies

The project uses Python 3.10 and the following libraries:

* TensorFlow + TensorBoard
* NumPy
* scikit-learn
* matplotlib + seaborn
* librosa
* plotly
* tqdm
* pydot / pydotplus / graphviz
* nbformat
* black + ruff (formatting/linting)
* pre-commit
* ipykernel


### Installing

Clone the repository:
git clone https://github.com/M1nX777/AML-speech-command-recognition.git
git checkout project-1

Install the dependencies using the project dependency file.
pipenv install
pipenv sync
pipenv shell

Running with Docker
Note: If Docker is not installed, install it first. A computer restart may be required after installation before Docker is available.
# Navigate to the Docker folder
cd docker

# Build the Docker image
docker build -t fastapi-app .

# Run the container
docker run -p 80:80 fastapi-app

Then open your browser and go to:
http://localhost:80


## Current Models

### Mel-Spectrogram + CNN

The CNN includes:

* Conv2D layers
* AveragePooling2D
* BatchNormalization
* GlobalMaxPooling2D
* Concatenation of convolution blocks
* Dense layer
* Dropout
* Softmax output layer with 8 classes

The optimizer used is AdamW with a learning rate of 1e-4.

### MFCC + MLP Baseline

The MLP baseline uses MFCC features:

* BatchNormalization
* Dense layers
* GlobalMaxPooling1D
* Dropout
* Softmax output layer with 8 classes
 
The optimizer used in the notebook is AdamW with a learning rate of 1e-5.

## Evaluation

The project evaluates the models using:
* Accuracy
* Per-class precision
* Per-class recall
* Per-class F1-score
* Classification report
* Training and validation loss curves
* Training and validation accuracy curves

## Authors

Group 13:

* Miruna Lungu
* Andrejs Tupikins
* Prayer Aguebor
* Petra Nicoara
