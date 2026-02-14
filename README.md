# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
### Step1
<br>

### Step2
<br>

### Step3
<br>

### Step4
<br>

### Step5
<br>

## Program:
```

Developed by: CHARUMATHI A
RegisterNumber: 212224230045
# Linear Regression using sklearn

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

# Load dataset (use diabetes dataset instead of removed Boston dataset)
data = datasets.load_diabetes()

# Define feature matrix (X) and target vector (y)
X = data.data
y = data.target

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1)

# Create linear regression object
reg = linear_model.LinearRegression()

# Train the model
reg.fit(X_train, y_train)

# Print coefficients
print("Coefficients:", reg.coef_)

# Print variance score (R² score)
print("Variance score: {:.2f}".format(reg.score(X_test, y_test)))

# Set plot style
plt.style.use('fivethirtyeight')

# Plot residual errors for training data
plt.scatter(reg.predict(X_train),
            reg.predict(X_train) - y_train,
            color="green", s=10, label="Train data")

# Plot residual errors for test data
plt.scatter(reg.predict(X_test),
            reg.predict(X_test) - y_test,
            color="blue", s=10, label="Test data")

# Plot zero error line
plt.hlines(y=0,
           xmin=0,
           xmax=max(reg.predict(X_test)),
           linewidth=2)

# Add legend and title
plt.legend(loc="upper right")
plt.title("Residual Errors")

# Show plot
plt.show()






```
## Output:
<img width="939" height="679" alt="image" src="https://github.com/user-attachments/assets/2269f542-4c98-44f3-81d1-83d99782b770" />


## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
