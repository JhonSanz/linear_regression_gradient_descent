""" Linear regression with gradient descent """

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import fmin_tnc


def hθ(theta, X):
    """ Straight line hypotesis, to describe our data

    Parameters:
        theta (vector): Parameters verctor
        X (matrix): Training examples data

    Returns:
        (vector): X * theta vector
    """
    return np.dot(X, theta)


def J(theta, X, y):
    """ Cost function J, called squared error function

    Parameters:
        theta (vector): Parameters verctor
        X (matrix): Training examples data
        y (vector): Right answers for our data

    Returns:
        (number): Squared error
    """
    cost = (1/2*len(y)) * np.sum(np.square(hθ(theta, X) - y))
    return cost


def derivated_term_J(theta, X, y):
    """ Squared error function derivated

    Parameters:
        theta (vector): Parameters verctor
        X (matrix): Training examples data
        y (vector): Right answers for our data

    Returns:
        (vector): Tentative parameters theta
    """
    θ = 1/len(y) * (np.dot(X.T, (hθ(theta, X) - y)))
    return θ


def gradient_descent(theta, X, y, α, iterations):
    """ Gradient descent algorithm

    Parameters:
        theta (vector): Parameters verctor
        X (matrix): Training examples data
        y (vector): Right answers for our data
        iterations (number): Number of iterations to converge

    Returns:
        theta (vector): Parameters theta found
        cost_history (vector): History of J values for each iteration
    """
    cost_history = np.zeros(iterations)
    for i in range(iterations):
        theta = theta - α * derivated_term_J(theta, X, y)
        cost_history[i] = J(theta, X, y)

    return theta, cost_history


def plot_results(theta, data):
    """ Plotting results, our straight line and data examples

    Parameters:
        theta (vector): Parameters verctor
    """
    data.plot.scatter(x="Attack", y="Speed")
    x = np.linspace(-0.4, 0.6, 1000)
    plt.plot(x, theta[0] + theta[1] * x + theta[2] * x ** 2,
             'r-')
    plt.show()


def gradient_descent_debugger(iterations, cost_history):
    """ Plotting history for each iteration, it helps to know
        what is happening with our algorithm

    Parameters:
        iterations (number): Number of iterations to converge
        cost_history (vector): History of J values for each iteration
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_ylabel('J(θ)')
    ax.set_xlabel('Iterations')
    ax.plot(range(iterations), cost_history, 'b.')
    plt.show()


def pro_min_functions(theta, X, y):
    """ Python minimizing tool kit <3 """
    return fmin_tnc(
        func=J, x0=theta, fprime=derivated_term_J,
        args=(X, y))[0]


def mean_normalization(described, example, col):
    return ((example - described.at["mean", col]) /
            (described.at["max", col] - described.at["min", col]))


def feature_scaling(data):
    """ scaling features speedup optimization """
    described = data.describe()
    for col in ["Attack", "Speed"]:
        for row in range(len(data[str(col)])):
            data.at[row, col] = mean_normalization(
                described, data.at[row, col], col)
    return data


def first_way():
    """ Used gradient descent and polinomial regression """
    data = pd.read_csv("pokemon_data.csv")

    data["Attack2"] = data["Attack"] ** 2
    data["bias"] = 1

    X = data[["bias", "Attack", "Attack2"]].values
    y = data["Speed"]

    """ I used a very very small α because it
        was overshooting the minimum """
    α = 0.0000000001
    y = y.values.reshape(len(y.values), 1)
    theta = np.array([0, 0, 0]).reshape(3, 1)
    iterations = 900000

    theta, cost_history = gradient_descent(
        theta, X, y, α, iterations)
    gradient_descent_debugger(iterations, cost_history)
    plot_results(theta, data)


def second_way():
    """ Used fmin_tnc, polinomial regression and feature scaling """
    data = pd.read_csv("pokemon_data.csv")
    data['Attack'] = pd.to_numeric(data['Attack'], downcast="float")
    data['Speed'] = pd.to_numeric(data['Speed'], downcast="float")

    data = feature_scaling(data)
    data["Attack2"] = data["Attack"] ** 2
    data["bias"] = 1

    X = data[["bias", "Attack", "Attack2"]]
    y = data["Speed"]
    theta = np.zeros(3)

    theta = pro_min_functions(theta, X, y)
    plot_results(theta, data)


if __name__ == "__main__":
    """ Be aware, if you want to run this code, you will need to change
        plot_results function
    """
    second_way()
