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

The project uses Python and the following libraries or modules in the current notebook:

* TensorFlow
* Keras
* NumPy
* pandas
* scikit-learn
* matplotlib
* seaborn
* librosa
* plotly
* tqdm

### Installing

Clone the repository:
git clone https://github.com/M1nX777/AML-speech-command-recognition.git

Install the dependencies using the project dependency file.

### Executing program

The program can be executed by using the notebook project_analysis_2.ipynb.

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
