# Linear Regression from Scratch

This repository contains a simple implementation of **Linear Regression** from scratch using **NumPy**. The model is trained using gradient descent to fit a line to the data.

## Description

Linear Regression is a supervised machine learning algorithm used for regression tasks, where the goal is to predict a continuous value. This implementation uses the gradient descent optimization technique to find the best-fitting line that minimizes the cost function (mean squared error).

## Files

- `linear_regression.py`: Contains the `LinearRegression` class with methods for training the model (`fit`), predicting values (`predict`), and calculating the cost function (`cost_function`).
  
## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/linear-regression-from-scratch.git
   ```

2. Import the `LinearRegression` class:

   ```python
   from linear_regression import LinearRegression
   ```

3. Create an instance of the `LinearRegression` class:

   ```python
   model = LinearRegression(lr=0.0001, epochs=100)
   ```

4. Fit the model with your training data (`X` as features and `y` as labels):

   ```python
   model.fit(X_train, y_train)
   ```

5. Make predictions:

   ```python
   y_pred = model.predict(X_test)
   ```

## Methods

### `__init__(self, lr=0.0001, epochs=100)`
- Initializes the learning rate (`lr`) and the number of epochs (`epochs`).

### `cost_function(self, y, y_prediction, m)`
- Computes the cost (mean squared error) between the true values (`y`) and predicted values (`y_prediction`).

### `fit(self, X, y)`
- Trains the linear regression model using gradient descent on the training data (`X` and `y`).

### `predict(self, X)`
- Predicts the output for new data (`X`).

## Example

```python
import numpy as np
from linear_regression import LinearRegression

# Example data
X_train = np.array([[1], [2], [3], [4], [5]])
y_train = np.array([[1], [2], [3], [4], [5]])

# Create and train the model
model = LinearRegression(lr=0.01, epochs=1000)
model.fit(X_train, y_train)

# Predict on new data
X_test = np.array([[6]])
y_pred = model.predict(X_test)

print(f"Prediction: {y_pred}")
```

## Requirements

- Python 3.x
- NumPy

Install dependencies:

```bash
pip install numpy
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
