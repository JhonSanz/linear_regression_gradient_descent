""" Linear regression with gradient descent """

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def hθ(theta, X):
    """ Straight line hypotesis, to describe our data

    Parameters:
        theta (vector): Parameters verctor
        X (matrix): Training examples data
    
    Retunrs:
        (vector): X * theta vector
    """
    return np.dot(X, theta)

def J(theta, X, y):
    """ Cost function J, called squared error function

    Parameters:
        theta (vector): Parameters verctor
        X (matrix): Training examples data
        y (vector): Right answers for our data

    Retunrs:
        (number): Squared error
    """
    cost = (1/2*len(y)) * np.sum(np.square(np.dot(X, theta)-y))
    return cost

def derivated_term_J(theta, X, y, α):
    """ Squared error function derivated

    Parameters:
        theta (vector): Parameters verctor
        X (matrix): Training examples data
        y (vector): Right answers for our data
        α (number): Learning rate

    Retunrs:
        (vector): Tentative parameters theta
    """
    θ =  theta - 1/len(y) * α * (X.transpose().dot(hθ(theta, X) - y))
    return θ

def gradient_descent(theta, X, y, α, iterations):
    """ Gradient descent algorithm

    Parameters:
        theta (vector): Parameters verctor
        X (matrix): Training examples data
        y (vector): Right answers for our data
        iterations (number): Number of iterations to converge

    Retunrs:
        theta (vector): Parameters theta found
        cost_history (vector): History of J values for each iteration
    """
    cost_history = np.zeros(iterations)
    for i in range(iterations):
        theta = derivated_term_J(theta, X, y, α)
        cost_history[i]  = J(theta, X, y)

    return theta, cost_history

def plot_results(size, theta):
    """ Plotting results, our straight line and data examples

    Parameters:
        theta (vector): Parameters verctor
        size (number): Max size to plot straight line
    """
    data.plot.scatter(x="Attack", y="Speed")
    plt.plot(np.arange(max(size)),
             theta[0] + theta[1] * np.arange(max(size)),
             'r-')
    plt.show()

def gradient_descent_debugger(iterations, cost_history):
    """ Plotting history for each iteration, it helps to know
        what is happening with our algorithm

    Parameters:
        iterations (number): Number of iterations to converge
        cost_history (vector): History of J values for each iteration
    """
    fig,ax = plt.subplots(figsize=(12,8))
    ax.set_ylabel('J(θ)')
    ax.set_xlabel('Iterations')
    ax.plot(range(iterations), cost_history, 'b.')
    plt.show()


if __name__ == "__main__":
    data = pd.read_csv("pokemon_data.csv")

    data["bias"] = 1

    X = data[["bias", "Attack"]].values
    y = data["Speed"]

    """ I used a very very small α because it
        was overshooting the minimum """
    α = 0.00001
    y = y.values.reshape(len(y.values), 1)
    theta = np.array([0, 0]).reshape(2, 1)
    iterations = 1500

    theta, cost_history = gradient_descent(
        theta, X, y, α, iterations)

    gradient_descent_debugger(iterations, cost_history)
    plot_results(max(y), theta)
