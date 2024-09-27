# Cancer Prediction Classification Model Performance Analysis

## Project Overview
This project based on Machine Learning  compares the performance of five different classification algorithms applied to a dataset with binary classification. The models evaluated are:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Support Vector Machine (SVM)
5. k-Nearest Neighbors (k-NN)

Each model was trained on the same dataset, and their performances were evaluated using key metrics such as **accuracy**, **precision**, **recall**, **F1-score**, and **confusion matrix**. Additionally, cross-validation was performed for the **Random Forest Classifier** to assess its consistency across multiple folds.

## Table of Contents
1. [Dataset Overview](#dataset-overview)
2. [Models Used](#models-used)
3. [Performance Metrics](#performance-metrics)
4. [Confusion Matrices](#confusion-matrices)
5. [Random Forest Cross-Validation](#random-forest-cross-validation)
6. [Conclusions](#conclusions)
9. [Contributing](#contributing)

    ```

## Dataset Overview
The dataset used in this project is related to binary classification (e.g., cancer detection, spam detection, etc.). The features and target are structured as follows:

- **Input Features**: Numeric or categorical data used to predict the target class.
- **Target Variable**: Binary output (0 or 1).

For more detailed information about the dataset, refer to the `data/` folder.

## Models Used
We used five popular machine learning algorithms for classification:

1. **Logistic Regression**: A linear model that estimates the probability of an instance belonging to a class.
2. **Decision Tree Classifier**: A non-linear model that splits data based on feature importance.
3. **Random Forest Classifier**: An ensemble method using multiple decision trees to reduce overfitting and increase accuracy.
4. **Support Vector Machine (SVM)**: A model that finds the hyperplane that best separates the classes in a high-dimensional space.
5. **k-Nearest Neighbors (k-NN)**: A simple model that classifies based on the majority vote of the nearest neighbors.

## Performance Metrics
The performance of each model was evaluated on both the **train** and **test** datasets. Below is a summary of key metrics:

| Model                   | Train Accuracy | Test Accuracy | Precision | Recall | F1 Score |
|--------------------------|----------------|---------------|-----------|--------|----------|
| **Logistic Regression**   | 0.9495         | 0.9211        | 0.93      | 0.93   | 0.93     |
| **Decision Tree**         | 0.9253         | 0.9123        | 0.93      | 0.93   | 0.93     |
| **Random Forest**         | 0.9253         | 0.9123        | 0.93      | 0.93   | 0.93     |
| **SVM**                   | 0.9253         | 0.9123        | 0.93      | 0.93   | 0.93     |
| **k-NN**                  | 0.9253         | 0.9123        | 0.93      | 0.93   | 0.93     |

## Confusion Matrices
Below are the confusion matrices for each model, highlighting the number of true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN).

- **Logistic Regression**
  ```
  [[41  5]
   [ 4 64]]
  ```

- **Decision Tree**
  ```
  [[40  5]
   [ 5 64]]
  ```

- **Random Forest**
  ```
  [[40  5]
   [ 5 64]]
  ```

- **Support Vector Machine (SVM)**
  ```
  [[40  5]
   [ 5 64]]
  ```

- **k-Nearest Neighbors (k-NN)**
  ```
  [[40  5]
   [ 5 64]]
  ```

## Random Forest Cross-Validation
We performed 5-fold cross-validation for the Random Forest model to evaluate its consistency. Below are the accuracy scores for each fold and the mean accuracy:

| Fold | Accuracy   |
|------|------------|
| 1    | 0.9341     |
| 2    | 0.9011     |
| 3    | 0.9121     |
| 4    | 0.9451     |
| 5    | 0.9341     |

**Mean Cross-Validation Accuracy**: 0.9253

## Conclusions
- **Logistic Regression** achieved the highest accuracy on the training data but showed a slight dip in test accuracy.
- **Decision Tree**, **Random Forest**, **SVM**, and **k-NN** had almost identical performances across all metrics, suggesting they are equally effective for this task.
- **Random Forest** demonstrated strong consistency with cross-validation, making it a robust choice for this dataset.
- All models had balanced precision, recall, and F1-scores, with minimal misclassification errors as observed in the confusion matrices.

## Contributing
Contributions are welcome! If you'd like to improve the models or suggest new approaches, please open an issue or submit a pull request.

---




 