# Online Shoppers Purchasing Intention: ML Classification Dashboard

## 1. Problem Statement

The objective of this project is to build and compare multiple machine-learning classification models that predict whether an online shopping session will result in a purchase. The project covers the end-to-end workflow: preparing the dataset, training and evaluating classifiers, saving trained models, and presenting their performance through an interactive Streamlit web application.

The target is **`Revenue`**:
- `True` (1): the online session resulted in a purchase.
- `False` (0): the online session did not result in a purchase.

The application loads the saved models and evaluates an uploaded labelled test CSV. It does **not** retrain the models on uploaded test data.

## 2. Dataset Description

This project uses the **Online Shoppers Purchasing Intention Dataset** from the UCI Machine Learning Repository.

The dataset contains **12,330 online shopping sessions** and **17 input features** describing visitor behaviour and session characteristics. These include page visits and time spent on administrative, informational and product-related pages, bounce and exit rates, page values, traffic type, visitor type, month, weekend, and other session attributes. The target column, `Revenue`, indicates whether a purchase was made during the session.

- **Dataset source:** [UCI Machine Learning Repository – Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)
- **Target variable:** `Revenue`
- **Problem type:** Binary classification

## 3. GitHub Repository Link

**GitHub Repository:**  https://github.com/2026aj05418/online-shoppers-classification.git

<!-- Add the GitHub repository URL here when available. -->

## 4. Models Used and Evaluation

The following five classification algorithms were implemented and evaluated on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The models are compared using Accuracy, Area Under the ROC Curve (AUC), Precision, Recall, F1-score and Matthews Correlation Coefficient (MCC).

### 4.1 Model Comparison Table

The following results are taken from the comparison of the 5 models. Values are shown as scores between 0 and 1.

| ML Model | Accuracy | AUC | Precision | Recall | F1-score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.847932 | 0.907823 | 0.505517 | 0.792453 | 0.617269 | 0.548885 |
| Decision Tree | 0.853933 | 0.931821 | 0.516129 | 0.897275 | 0.655311 | 0.606186 |
| K-Nearest Neighbors | 0.900487 | 0.940862 | 0.762934 | 0.517820 | 0.616922 | 0.576085 |
| Gaussian Naive Bayes | 0.262368 | 0.777007 | 0.172813 | 0.994759 | 0.294469 | 0.142654 |
| Support Vector Machine | 0.883131 | 0.924779 | 0.590539 | 0.798218 | **0.678850** | **0.619699** |

### 4.2 Observations on Model Performance

| ML Model | Observation |
|---|---|
| **Logistic Regression** | Achieved an accuracy of 0.847932 and AUC of 0.907823. Its recall (0.792453) is considerably higher than its precision (0.505517), indicating that it identifies a substantial proportion of purchase sessions but also produces false-positive predictions. |
| **Decision Tree** | Achieved a recall of 0.897275, the second-highest F1-score (0.655311), and an MCC of 0.606186. It identifies purchase sessions effectively, although its precision is comparatively modest at 0.516129. |
| **K-Nearest Neighbors** | Recorded the highest accuracy (0.900487), AUC (0.940862), and precision (0.762934). However, its recall is 0.517820, so it misses a larger proportion of actual purchase sessions than the other models with higher recall. |
| **Gaussian Naive Bayes** | Recorded very high recall (0.994759), but its precision (0.172813), accuracy (0.262368), F1-score (0.294469), and MCC (0.142654) are low. This indicates that it predicts the purchase class for many sessions, resulting in a large number of false positives. |
| **Support Vector Machine** | Achieved the highest F1-score (0.678850) and MCC (0.619699), with accuracy of 0.883131 and recall of 0.798218. It provides the strongest balance between precision and recall among the five models according to these results. |

### 4.3 Overall Winner

**Support Vector Machine (SVM)** is the overall winner based on the highest F1-score (0.678850) and MCC (0.619699). These metrics indicate the best overall balance and classification correlation among the evaluated models in this comparison.

The best model can depend on the business objective. For example, if identifying as many actual purchase sessions as possible is the priority, the Decision Tree's higher recall may be useful. If avoiding false-positive purchase predictions is more important, KNN's higher precision may be preferable.

## 5. Streamlit Application

The interactive Streamlit application provides the following functionality:

- **Dataset overview:** Displays a description of the dataset, target variable, dataset summary and feature information.
- **CSV upload:** Allows a user to upload labelled test data in CSV format.
- **Model selection:** Provides a dropdown to select and evaluate an individual classifier.
- **Evaluation metrics:** Displays Accuracy, AUC, Precision, Recall, F1-score and MCC.
- **Confusion matrix and classification report:** Shows class-wise prediction performance for the selected model.
- **Predicted vs. actual:** Displays row-level actual and predicted labels, purchase probabilities where available, and whether each prediction is correct.
- **Model comparison:** Compares all five classifiers and provides metric-based rankings and visual comparisons.

The uploaded test CSV must contain all required input features and the `Revenue` target column. The application uses the saved trained models to generate predictions and calculate evaluation metrics. It does not train or fit a model using the uploaded test data.

**Live Streamlit App:**  https://2026aj05418-bits-ml-assignment2.streamlit.app

<!-- Add the deployed Streamlit Community Cloud URL here when available. -->

## 6. Repository Structure

The repository contains the following files and folders:

```text
project-folder/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── data/
│   └── online_shoppers_intention.csv
├── test_data.csv
└── model/
    ├── metadata.json
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── gaussian_nb.joblib
    └── svm.joblib
```


## 7. Running the Application Locally

1. Clone the repository:

   ```bash
   git clone <YOUR_GITHUB_REPOSITORY_URL>
   cd <YOUR_PROJECT_FOLDER>
   ```

2. Create and activate a Python virtual environment (recommended).

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the Streamlit application:

   ```bash
   streamlit run app.py
   ```

5. Open the local URL displayed in the terminal. Upload a labelled test CSV, select a model, and review the evaluation results.

---

*Project prepared as part of the M.Tech (AIML), Work Integrated Learning Programmes Division, BITS Pilani, Machine Learning Assignment 2.*
