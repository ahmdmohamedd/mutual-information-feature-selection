# Mutual Information Feature Selection

## Overview
This repository demonstrates a feature selection method using mutual information, implemented in MATLAB. The goal of this project is to improve model performance by selecting the most relevant features from a dataset. It includes an evaluation of a multi-class Support Vector Machine (SVM) classifier to illustrate the effectiveness of the selected features.

## Contents
- `feature_selection.m`: Main script for performing feature selection and model evaluation.
- `mutual_information.m`: Function to compute the mutual information between features and labels.

## Dataset
For demonstration purposes, the project utilizes the **UCI Human Activity Recognition (HAR)** dataset. You can download it from [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones).

### Dataset Structure
- `train/X_train.txt`: Training data features.
- `train/y_train.txt`: Training data labels.

Ensure that the dataset is organized as follows in your working directory:
```
UCI_HAR_Dataset/
├── train/
│   ├── X_train.txt
│   └── y_train.txt
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ahmdmohamedd/mutual-information-feature-selection.git
   cd mutual-information-feature-selection
   ```
2. Open the `feature_selection.m` script in MATLAB.

## Usage
1. Make sure you have the necessary toolboxes installed in MATLAB.
2. Run the `feature_selection.m` script:
   ```matlab
   feature_selection
   ```
3. The script will output the accuracy, precision, recall, and F1-score of the model after performing feature selection.

## Evaluation Metrics
- **Accuracy**: Proportion of true results among the total number of cases examined.
- **Precision**: Ratio of correctly predicted positive observations to the total predicted positives.
- **Recall**: Ratio of correctly predicted positive observations to the all observations in actual class.
- **F1-Score**: Weighted average of precision and recall.

## Results
The output of the model will provide:
- Accuracy
- Macro-Average Precision
- Macro-Average Recall
- Macro-Average F1-Score

## Future Work
- Implement additional feature selection methods for comparison.
- Explore other classification algorithms and evaluate their performance.
- Integrate visualization techniques to better illustrate the model's performance and feature importance.

## Acknowledgments
- The UCI Machine Learning Repository for the dataset.
- MATLAB documentation for the functions used.
```
