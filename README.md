# Nepal Weather Prediction and Climate Analysis

A data science project analyzing Nepal's climate data from 1981 to 2019 across 62 districts to predict rainfall using statistical analysis, machine learning models, and deep learning.

## Project Files

```
Nepal-Weather-Prediction/
|-- dataset/
|   +-- dailyclimate.zip                      # Compressed climate data (extract to dailyclimate.csv)
|-- 01_Nepal_Weather_Statistics.ipynb         # Descriptive statistics and outlier analysis
|-- 02_Nepal_Weather_Machine_Learning.ipynb   # ML pipeline, models, and evaluation
|-- 03_Nepal_Weather_Deep_Learning.ipynb      # Perceptron, activations, and ANN training
+-- README.md
```

## Notebooks Summary

### 01_Nepal_Weather_Statistics.ipynb
We explore the dataset and calculate basic descriptive statistics including mean, median, mode, variance, standard deviation, and range. We also check distribution shapes (skewness and kurtosis), plot the correlation heatmap, and detect outliers using IQR and Z-score methods.

### 02_Nepal_Weather_Machine_Learning.ipynb
We clean the data, handle outliers using IQR capping, and scale features using StandardScaler. We train Logistic Regression, KNN, Decision Tree, and Random Forest models to predict next-day rain. We also tune the decision tree with GridSearchCV, run 5-fold cross-validation, and compare performance with confusion matrices and ROC curves.

### 03_Nepal_Weather_Deep_Learning.ipynb
We demonstrate deep learning concepts including a simple Perceptron from scratch and plots for Sigmoid, ReLU, and Tanh activation functions. We then train a Multi-Layer Neural Network (ANN) using Adam optimizer to predict rainfall, track loss and accuracy curves across training iterations, and compare it with the machine learning models.

## Results Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.8124 | 0.6582 | 0.3211 | 0.4318 | 0.7981 |
| K-Nearest Neighbors (k=7) | 0.8193 | 0.6274 | 0.4533 | 0.5263 | 0.8142 |
| Decision Tree (Tuned) | 0.8285 | 0.6710 | 0.4589 | 0.5451 | 0.8230 |
| Random Forest | 0.8395 | 0.7180 | 0.4720 | 0.5698 | 0.8872 |
| Artificial Neural Network (ANN) | 0.8348 | 0.7012 | 0.4680 | 0.5610 | 0.8750 |

Random Forest achieved the best overall score (~84% accuracy, 0.887 ROC-AUC), followed closely by the ANN.

## How to Run

Install required packages:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

Open and run each notebook in Jupyter:
- `01_Nepal_Weather_Statistics.ipynb`
- `02_Nepal_Weather_Machine_Learning.ipynb`
- `03_Nepal_Weather_Deep_Learning.ipynb`
