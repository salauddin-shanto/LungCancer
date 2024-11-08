# Lung Cancer Histopathological Image Classification Using CNN

This project implements a deep learning model using **Convolutional Neural Networks (CNNs)** to classify lung cancer histopathological images. The dataset consists of images categorized into three classes: **lung adenocarcinoma (lung_aca)**, **lung squamous cell carcinoma (lung_scc)**, and **normal lung tissue (lung_n)**. The model is trained using **5-fold cross-validation** and includes techniques to prevent overfitting, such as **data augmentation** and **early stopping**.

## Table of Contents

1. [Overview](#Overview)
2. [Dataset](#Dataset)
3. [Model Architecture](#Model-Architecture)
4. [Data Preprocessing](#Data-Preprocessing)
5. [Cross-Validation](#Cross-Validation)
6. [Performance Metrics](#Performance-Metrics)
7. [Results and Visualization](#Results-and-Visualization)
8. [How to Run the Notebook](#How-to-Run-the-Notebook)
9. [License](#License)

## Overview

Lung cancer is one of the leading causes of cancer-related deaths worldwide. Histopathological analysis plays a crucial role in diagnosing lung cancer, but manual classification can be time-consuming and prone to human error. In this project, we use **deep learning** to automate the classification of lung cancer subtypes from histopathological images. The model is trained using **5-fold cross-validation**, and several evaluation metrics are employed to assess its performance.

## Dataset

The dataset used in this project is sourced from **Kaggle**, specifically the **Lung and Colon Cancer Histopathological Images** dataset. It contains histopathological images labeled into three categories:

- **lung_aca**: Lung adenocarcinoma (cancerous tissue)
- **lung_scc**: Lung squamous cell carcinoma (cancerous tissue)
- **lung_n**: Normal lung tissue

The images are organized in directories, with each directory corresponding to a class. 

The dataset could be found in https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images
## Model Architecture

The classification model is a **Convolutional Neural Network (CNN)** with the following architecture:

1. **Input Layer**: The images are resized to a uniform shape (224x224x3).
2. **Convolutional Layers**: Multiple layers of 2D convolution with ReLU activation functions.
3. **Max Pooling**: Applied after each convolutional layer to downsample the feature maps.
4. **Fully Connected Layers**: After flattening the features, fully connected layers are applied to make the final prediction.
5. **Dropout**: To prevent overfitting, dropout layers are added between fully connected layers.
6. **Output Layer**: A softmax activation function is used to classify the images into three categories.

### Libraries Used

- **TensorFlow** and **Keras** for building and training the CNN model.
- **Matplotlib** and **Seaborn** for data visualization.
- **Scikit-learn** for cross-validation and performance metrics.
- **PIL** for image processing.

## Data Preprocessing

To ensure that the model can generalize well to unseen data, the following preprocessing steps were applied:

- **Resizing**: All images were resized to 224x224 pixels.
- **Normalization**: Pixel values were normalized to the range [0, 1].
- **Data Augmentation**: Techniques like rotation, flipping, and zooming were applied to increase the variety of training data and prevent overfitting.

## Cross-Validation

To evaluate the model's performance, **5-fold cross-validation** was employed. This approach splits the dataset into 5 subsets and trains the model on 4 subsets while testing on the remaining one. The results from all 5 folds are then averaged to provide a robust estimate of the model's performance.

## Performance Metrics

The following metrics were calculated to assess the performance of the model:

- **Accuracy**: Percentage of correctly classified images.
- **Precision**: The proportion of positive predictions that are actually correct.
- **Recall**: The proportion of actual positive cases that are correctly identified.
- **F1-Score**: The harmonic mean of precision and recall.
- **AUC-ROC Curve**: The area under the receiver operating characteristic curve, which provides insight into the classifier's ability to distinguish between the classes.

## Results and Visualization

- **Countplot**: Visualizes the distribution of the dataset across the three classes.   ![countplot](https://github.com/user-attachments/assets/c6e67138-ea16-4db7-b1b3-47a4746078d2)
- **Confusion Matrix**: Provides insights into the true positive, false positive, true negative, and false negative rates. ![confusion_matrix](https://github.com/user-attachments/assets/b314482e-f5d2-4d15-91be-2add709c3132)
- **ROC Curve**: Displays the tradeoff between true positive rate and false positive rate for each class. ![roc_curve](https://github.com/user-attachments/assets/e42a820c-64e1-466c-9d41-cc00d0001667)
- **Accuracy and Loss Curves**: Shows the training and validation accuracy and loss over epochs. ![avg_training_validation_accuracy](https://github.com/user-attachments/assets/64ba5578-a553-455f-9ad3-0973408636e2)  ![avg_training_validation_loss](https://github.com/user-attachments/assets/1fecac11-93d8-4719-abef-a49742f65a3f)

Sample plots are saved as image files and can be found in the output directory of the notebook.




## How to Run the Notebook

To run the notebook and replicate the results:

1. **Clone or Download the Repository**:
   You can clone this repository to your local machine using Git, or you can download the notebook directly from Kaggle.

   ```bash
   https://github.com/salauddin-shanto/LungCancer.git
   
2. **Set Up Your Environment: Install the required dependencies using pip:**
      ```bash
       pip install -r requirements.txt
  
3. **Run the Notebook**: Open the lung-colon-cancer0.ipynb notebook in Kaggle or Jupyter Notebook, and execute the cells to run the model training and evaluation.

4. **Examine the Results**: The notebook will output the training and validation accuracy and loss for each fold, confusion matrix, ROC curve, and other relevant plots to assess the model's performance.

