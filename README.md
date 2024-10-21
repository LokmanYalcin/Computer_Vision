# Weather Classification Using Fine-Tuning of DenseNet121

## Project Overview

This project focuses on classifying weather conditions based on images using deep learning techniques. By utilizing transfer learning, particularly the **DenseNet121** model, we aim to accurately categorize weather images into one of four classes: Cloudy, Sunshine, Rainy, and Sunrise. Two different approaches were tested: **Feature Extraction** and **Fine-Tuning**.

## Dataset

The **MWD (Multi-class Weather Dataset)** was used for this project. It consists of **1125 images**, distributed across four classes:
- **Cloudy**: 300 images
- **Sunshine**: 235 images
- **Rainy**: 215 images
- **Sunrise**: 357 images

The dataset was divided into training and testing sets, with a validation set created from the testing data.

## Methodology

The model was trained using the **DenseNet121** architecture pre-trained on ImageNet. We compared two approaches:
1. **Feature Extraction**: The pre-trained convolutional base of DenseNet121 was frozen, and a custom classifier was trained on top.
2. **Fine-Tuning**: After initial feature extraction, we unfroze the last layers of the base model and retrained them along with the classifier for better performance.

Both approaches utilized the **softmax activation function** in the final dense layer for multi-class classification.

## Results

The **Fine-Tuning** approach achieved significantly better performance, with:
- **Accuracy**: 97%
- **Precision**: 97%
- **Recall**: 97%
- **F1-score**: 97%

The training was done using **TensorFlow** and **Keras** libraries on **Google Colab**, with GPU support for accelerated training.

## Evaluation Metrics

- **Accuracy**: Proportion of correctly classified images.
- **Precision**: Accuracy of positive predictions.
- **Recall**: True positive rate.
- **F1-Score**: Harmonic mean of precision and recall.

The model achieved a near-perfect accuracy of **97%** across all metrics, showing significant improvement after fine-tuning.

## Model Architecture

The model utilizes the **DenseNet121** backbone for feature extraction and fine-tuning. Techniques such as **Dropout** and **Early Stopping** were applied for regularization, preventing overfitting and optimizing training efficiency.

## Visualizations

- **Confusion Matrix**: Demonstrates classification performance across the four classes.
- **Training and Validation Accuracy**: Plot showing performance improvement with fine-tuning.

## Conclusion

This project demonstrates the power of **transfer learning** and **fine-tuning** in image classification tasks. The model’s high accuracy and robustness make it suitable for real-world applications, including weather prediction systems, where accurate weather classification is essential for decision-making.

## Requirements

To replicate this project, you will need:
- Python 3.x
- TensorFlow 2.x
- Keras
- OpenCV
- Matplotlib
- NumPy
- Pandas


