import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

# load the data from the dataset
data = pd.read_csv('Honda_Data.csv')
print(data.head())

# deciding what should be my train and target data i will target  Close that i want to predict so i put it to the y axis
x_train = data[["Date", "High", "Open", "Volume"]]
y_train = data["Close"]  # Target is Close price

# print out the shape and first 5 rows just to see what im working with 
print("Type of x:", type(x_train))
print("First five elements of x are: \n", x_train[:5])
print("The shape of X is:", x_train.shape)

# print out target
print("Type of y_train:", type(y_train))
print("First five elements of y_train are: \n", y_train)
print("Number of training examples (m):", len(x_train))

# Convert 'Date' to ordinal values for use in regression
data["Date"] = pd.to_datetime(data["Date"])
data["Date"] = data["Date"].map(pd.Timestamp.toordinal)
x_train["Date"] = data["Date"].to_numpy()

# Standardize the features (Date, High, Open, Volume) so they all have similar scales
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)

# Initialize weights and bias
w = np.random.randn(x_train_scaled.shape[1]) * 0.01  # Small random initialization
b = 0

# learning rate alpha
alpha = 0.01
iterations = 1000
cost_history = []  # To store cost at each iteration

# Cost function (Mean Squared Error)
def compute_cost(X, y, w, b):
    m = X.shape[0]  # number of training examples
    f_wb = np.dot(X, w) + b  # predictions
    err = f_wb - y  # error
    cost = np.sum(err ** 2) / (2 * m)  # MSE
    return cost

# gradient descent function
def compute_gradient(X, y, w, b):
    m, n = X.shape  # number of training examples and features
    dj_dw = np.zeros_like(w)  # Initialize gradients to match the shape of w
    dj_db = 0  # Initialize bias gradient

    # Iterate over each training example
    for i in range(m):
        err = (np.dot(X[i], w) + b - y[i])  # Error term
        # Update gradients
        for j in range(n):
            dj_dw[j] += err * X[i, j]
        dj_db += err

    # Average gradients over all examples
    dj_dw /= m
    dj_db /= m

    return dj_db, dj_dw

# Gradient Descent
for i in range(iterations):
    dj_db, dj_dw = compute_gradient(x_train_scaled, y_train, w, b)  # Compute gradients
    w -= alpha * dj_dw  # Update weights
    b -= alpha * dj_db  # Update bias

    # Monitor cost every 100 iterations
    if i % 100 == 0:
        cost = compute_cost(x_train_scaled, y_train, w, b)
        cost_history.append(cost)
        print(f"Iteration {i}: Cost {cost}")

# Prediction function
def predict(X, w, b):
    return np.dot(X, w) + b

# Make predictions
predictions = predict(x_train_scaled, w, b)
print("First 5 predictions:", predictions[:5])
print("Actual 'Close' prices:", y_train[:5])

# Plot cost history over iterations
plt.plot(range(0, iterations, 100), cost_history)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost Function Over Iterations')
plt.show()
