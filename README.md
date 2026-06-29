 # Wine Quality Prediction using Random Forest Regression

## Overview

This project predicts the quality of wine using a Random Forest Regression model. The dataset contains various chemical properties of wine, and the model learns the relationship between these features and the wine quality score.

---

## Libraries Used

The following Python libraries are required:

```python
pandas
scikit-learn
```

Import statements:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
```

---

## Dataset

* Dataset Name: `winequality.csv`
* Target Variable: `quality`
* Features: All remaining columns representing wine characteristics.

---

## Project Workflow

### Step 1: Import Required Libraries

Import Pandas for data handling and Scikit-learn modules for model training and evaluation.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
```

---

### Step 2: Load the Dataset

Read the CSV file into a Pandas DataFrame.

```python
data = pd.read_csv("winequality.csv")
```

---

### Step 3: Separate Features and Target Variable

Split the dataset into:

* Input Features (`X`)
* Target Variable (`y`)

```python
X = data.drop("quality", axis=1)
y = data["quality"]
```

---

### Step 4: Split the Dataset

Divide the dataset into training and testing sets.

* Training Data: 80%
* Testing Data: 20%

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

### Step 5: Create the Random Forest Model

Initialize the Random Forest Regressor.

```python
rf = RandomForestRegressor(random_state=42)
```

---

### Step 6: Train the Model

Train the model using the training dataset.

```python
rf.fit(X_train, y_train)
```

---

### Step 7: Make Predictions

Generate predictions using the testing dataset.

```python
predictions = rf.predict(X_test)
```

---

### Step 8: Evaluate the Model

Calculate the R² Score to measure model performance.

```python
score = r2_score(y_test, predictions)
```

Display the result:

```python
print("R2 Score:", score)
```

---

## Evaluation Metric

### R² Score (Coefficient of Determination)

The R² Score indicates how well the model explains the variance in the target variable.

* Value close to 1 → Better performance
* Value close to 0 → Poor performance

---

## Output

The program displays the R² Score of the trained model.

Example:

```text
R2 Score: 0.52
```

(Note: Actual results may vary depending on the dataset used.)

---

## How to Run

1. Install required packages:

```bash
pip install pandas scikit-learn
```

2. Place `winequality.csv` in the project directory.

3. Run the Python script:

```bash
python main.py
```

4. View the generated R² Score in the terminal.

---

## Machine Learning Algorithm

**Random Forest Regressor**

Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy and reduce overfitting.

---

## Author

Project developed for learning and implementing Machine Learning using Python and Scikit-learn.





# Heart Disease Prediction using Logistic Regression

## Overview

This project predicts whether a patient has heart disease using the Logistic Regression algorithm. The dataset contains patient health-related information such as age, cholesterol level, blood pressure, and other medical attributes. The model is trained to classify patients into two categories: presence or absence of heart disease.

---

# Libraries Used

The following Python libraries are required:

```python
pandas
scikit-learn
```

Import statements:

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

---

# Dataset

* **Dataset Name:** `heart_disease.csv`
* **Target Variable:** `target`
* **Input Features:** All remaining columns containing patient medical information.

---

# Project Workflow

## Step 1: Import Required Libraries

Import all necessary libraries for data processing, model training, and performance evaluation.

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

---

## Step 2: Load the Dataset

Read the Heart Disease dataset into a Pandas DataFrame.

```python
df = pd.read_csv("heart_disease.csv")
```

Display the first five rows:

```python
df.head()
```

---

## Step 3: Encode Categorical Data

Convert the categorical **sex** column into numerical values using Label Encoding.

```python
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'].str.lower())
```

This step allows the machine learning model to process categorical information.

---

## Step 4: Separate Features and Target Variable

Split the dataset into:

* **Features (X)** – Independent variables
* **Target (y)** – Heart disease prediction

```python
x = df.drop('target', axis=1)
y = df['target']
```

---

## Step 5: Split the Dataset

Divide the dataset into training and testing sets.

* **Training Data:** 80%
* **Testing Data:** 20%

```python
xtrain, xtest, ytrain, ytest = train_test_split(
    x, y, test_size=0.2, random_state=42
)
```

---

## Step 6: Handle Missing Values

Replace missing values using the **Mean Imputation** technique.

```python
imputer = SimpleImputer(strategy='mean')

xtrain = imputer.fit_transform(xtrain)
xtest = imputer.transform(xtest)
```

This ensures that the dataset contains no missing values before training.

---

## Step 7: Create the Logistic Regression Model

Initialize the Logistic Regression classifier.

```python
model = LogisticRegression()
```

---

## Step 8: Train the Model

Train the model using the training dataset.

```python
model.fit(xtrain, ytrain)
```

---

## Step 9: Make Predictions

Predict heart disease for the testing dataset.

```python
ypred = model.predict(xtest)
```

---

## Step 10: Evaluate the Model

Calculate the prediction accuracy.

```python
accu = accuracy_score(ytest, ypred)

print("Accuracy :", accu)
```

---

# Evaluation Metric

## Accuracy Score

Accuracy measures the percentage of correctly classified samples.

**Formula:**

```text
Accuracy = (Correct Predictions / Total Predictions) × 100
```

Higher accuracy indicates better classification performance.

---

# Output

The program displays the prediction accuracy.

Example:

```text
Accuracy : 0.89
```

> **Note:** The actual accuracy may vary depending on the dataset and preprocessing.

---

# How to Run

### 1. Install Required Libraries

```bash
pip install pandas scikit-learn
```

### 2. Place the Dataset

Save `heart_disease.csv` in the project directory.

### 3. Run the Python Script

```bash
python main.py
```

### 4. View the Output

The model predicts heart disease and prints the **Accuracy Score** in the terminal.

---

# Machine Learning Algorithm

## Logistic Regression

Logistic Regression is a supervised machine learning algorithm used for **binary classification** problems. It estimates the probability that a given input belongs to one of two classes, making it suitable for predicting whether a patient has heart disease.

---

# Note

The provided code is missing the model initialization. Before training the model, add the following line:

```python
model = LogisticRegression(random_state=42)
```

Without this line, the code will raise a `NameError` because `model` has not been defined.

---

# Author

Project developed for learning and implementing Machine Learning classification using Python and Scikit-learn.









# House Price Prediction using Random Forest Regression

## Overview

This project predicts house prices using the **Random Forest Regression** algorithm. The model is trained on housing-related features such as property characteristics, location-based attributes, and other available factors to estimate the selling price of a house. The model's performance is evaluated using the **R² Score**, which measures how well the predicted values match the actual house prices.

---

# Libraries Used

The following Python libraries are required:

```python
pandas
scikit-learn
```

Import statements:

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
```

> **Note:** In this implementation, `LabelEncoder` and `SimpleImputer` are imported but are not used in the code.

---

# Dataset

* **Dataset Name:** `house_prices.csv`
* **Target Variable:** `price`
* **Input Features:** All remaining columns representing house characteristics.

---

# Project Workflow

## Step 1: Import Required Libraries

Import the necessary libraries for data handling, model training, and evaluation.

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
```

---

## Step 2: Load the Dataset

Read the house price dataset into a Pandas DataFrame.

```python
df = pd.read_csv("house_prices.csv")
```

Display the first five records.

```python
df.head()
```

---

## Step 3: Separate Features and Target Variable

Split the dataset into:

* **Features (X)** – House-related attributes used for prediction.
* **Target (y)** – House price.

```python
x = df.drop(['price'], axis=1)
y = df['price']
```

---

## Step 4: Split the Dataset

Divide the dataset into training and testing sets.

* **Training Data:** 80%
* **Testing Data:** 20%

```python
xtrain, xtest, ytrain, ytest = train_test_split(
    x, y, test_size=0.2, random_state=2
)
```

---

## Step 5: Create the Random Forest Regression Model

Initialize the Random Forest Regressor with **200 decision trees**.

```python
model = RandomForestRegressor(n_estimators=200)
```

Increasing the number of trees generally improves prediction stability, although it may increase training time.

---

## Step 6: Train the Model

Train the regression model using the training dataset.

```python
model.fit(xtrain, ytrain)
```

---

## Step 7: Make Predictions

Predict house prices for the testing dataset.

```python
ypred = model.predict(xtest)
```

---

## Step 8: Evaluate the Model

Evaluate the model using the **R² Score**.

```python
accu = r2_score(ytest, ypred)

print("R2 Score :- ", accu)
```

The R² Score measures how accurately the model predicts house prices.

---

# Evaluation Metric

## R² Score (Coefficient of Determination)

The **R² Score** indicates how well the regression model explains the variation in the target variable.

**Interpretation:**

* **1.0** → Perfect prediction
* **0.0** → Model performs no better than predicting the average value
* **Negative Value** → Model performs worse than a simple average prediction

Higher R² values indicate better model performance.

---

# Output

The program displays the R² Score of the trained model.

Example:

```text
R2 Score :- 0.89
```

> **Note:** The actual score depends on the dataset and may vary.

---

# How to Run

### 1. Install Required Libraries

```bash
pip install pandas scikit-learn
```

### 2. Place the Dataset

Save `house_prices.csv` in the project directory.

### 3. Run the Python Script

```bash
python main.py
```

### 4. View the Output

The terminal will display the **R² Score**, indicating the prediction performance of the model.

---

# Machine Learning Algorithm

## Random Forest Regression

Random Forest Regression is an ensemble learning algorithm that combines multiple decision trees to improve prediction accuracy. Each tree makes an independent prediction, and the final output is calculated by averaging the predictions of all trees. This approach helps reduce overfitting and provides more reliable results for regression problems.

---

# Project Summary

* ✔ Imported the required Python libraries
* ✔ Loaded the house price dataset
* ✔ Selected input features and target variable
* ✔ Split the dataset into training and testing sets
* ✔ Created a Random Forest Regression model
* ✔ Trained the model using the training data
* ✔ Predicted house prices for the testing data
* ✔ Evaluated model performance using the R² Score

---

# Future Improvements

* Perform data preprocessing to handle missing values.
* Encode categorical features using appropriate encoding techniques.
* Apply feature engineering to improve prediction accuracy.
* Tune hyperparameters using Grid Search or Randomized Search.
* Save the trained model using Pickle or Joblib for deployment.

---

# Author

Project developed for learning and implementing Machine Learning regression using Python and Scikit-learn.







# Diabetes Prediction using Support Vector Machine (SVM)

## Overview

This project predicts whether a person has diabetes using the Support Vector Machine (SVM) classification algorithm. The model is trained on medical information such as glucose level, blood pressure, BMI, insulin, and other health-related attributes. After training, both the model and the feature scaler are saved for future predictions.

---

# Libraries Used

The following Python libraries are required:

```python
pandas
numpy
scikit-learn
pickle
```

Import statements:

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
```

---

# Dataset

* **Dataset Name:** `diabetes.csv`
* **Target Variable:** `Outcome`
* **Input Features:** All remaining columns containing patient medical information.

---

# Project Workflow

## Step 1: Import Required Libraries

Import the necessary libraries for data manipulation, preprocessing, machine learning, model evaluation, and model saving.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
```

---

## Step 2: Load the Dataset

Read the diabetes dataset into a Pandas DataFrame.

```python
df = pd.read_csv("diabetes.csv")
```

Display the first five rows.

```python
df.head()
```

---

## Step 3: Data Cleaning

Remove missing values and duplicate records from the dataset.

```python
df = df.dropna()
df = df.drop_duplicates()
```

This helps improve the quality of the dataset before training the model.

---

## Step 4: Separate Features and Target Variable

Split the dataset into:

* **Features (X)** – Medical attributes used for prediction.
* **Target (y)** – Diabetes status (`Outcome`).

```python
X = df.drop(columns=['Outcome'], axis=1)
y = df['Outcome']
```

---

## Step 5: Split the Dataset

Divide the dataset into training and testing sets.

* **Training Data:** 80%
* **Testing Data:** 20%

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

---

## Step 6: Feature Scaling

Standardize the feature values using **StandardScaler**.

```python
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Feature scaling improves the performance of SVM by ensuring that all features have a similar scale.

---

## Step 7: Create the SVM Model

Initialize the Support Vector Machine classifier with a linear kernel.

```python
cf = svm.SVC(kernel='linear')
```

---

## Step 8: Train the Model

Train the SVM classifier using the scaled training data.

```python
cf.fit(X_train_scaled, y_train)
```

---

## Step 9: Make Predictions

Predict diabetes status for the testing dataset.

```python
x_test_pred = cf.predict(X_test_scaled)
```

---

## Step 10: Evaluate the Model

Calculate the model's prediction accuracy.

```python
x_test_acc = accuracy_score(y_test, x_test_pred)

print(x_test_acc)
print(x_test_acc * 100)
```

The first output shows the accuracy as a decimal value, while the second displays it as a percentage.

---

## Step 11: Save the Trained Model

Save the trained SVM model using the Pickle library.

```python
pickle.dump(cf, open("diabetes_model.pkl", "wb"))
```

The saved model can be loaded later for making predictions without retraining.

---

## Step 12: Save the Feature Scaler

Save the trained StandardScaler.

```python
pickle.dump(scaler, open("diabetes_scaler.pkl", "wb"))
```

Saving the scaler ensures that new input data is transformed in the same way as the training data before making predictions.

---

# Evaluation Metric

## Accuracy Score

Accuracy measures how many predictions made by the model are correct.

**Formula:**

```text
Accuracy = (Correct Predictions / Total Predictions) × 100
```

Higher accuracy indicates better classification performance.

---

# Output

The program displays:

* Model accuracy
* Accuracy percentage
* Confirmation message after saving the model and scaler

Example:

```text
0.81
81.0
Diabetes model and scaler saved successfully!
```

> **Note:** Actual accuracy may vary depending on the dataset used.

---

# Saved Files

After running the program, the following files are created:

| File                  | Description                      |
| --------------------- | -------------------------------- |
| `diabetes_model.pkl`  | Stores the trained SVM model     |
| `diabetes_scaler.pkl` | Stores the fitted StandardScaler |

These files can be loaded later to make predictions on new patient data.

---

# How to Run

### 1. Install Required Libraries

```bash
pip install pandas numpy scikit-learn
```

### 2. Place the Dataset

Save `diabetes.csv` in the project directory.

### 3. Run the Python Script

```bash
python main.py
```

### 4. View the Results

The program will:

* Clean the dataset
* Scale the features
* Train the SVM model
* Evaluate its performance
* Save the trained model and scaler as `.pkl` files

---

# Machine Learning Algorithm

## Support Vector Machine (SVM)

Support Vector Machine (SVM) is a supervised machine learning algorithm primarily used for classification tasks. It works by finding the optimal decision boundary (hyperplane) that best separates data into different classes. In this project, a **linear kernel** is used to classify patients as diabetic or non-diabetic.

---

# Project Summary

* ✔ Imported required libraries
* ✔ Loaded the diabetes dataset
* ✔ Cleaned the dataset by removing missing and duplicate values
* ✔ Split the data into features and target
* ✔ Divided the data into training and testing sets
* ✔ Standardized the features using `StandardScaler`
* ✔ Trained an SVM classifier
* ✔ Evaluated the model using Accuracy Score
* ✔ Saved both the trained model and scaler using Pickle

---

# Author

Project developed for learning and implementing Machine Learning classification using Python and Scikit-learn.









