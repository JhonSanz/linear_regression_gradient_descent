# Linear Regression with Gradient Descent
As a supervised learning algorithm, our dataset tell us how data examples behaviour, and our job is to fit a straight line that describe better all our data.

Once we have our straight line, we can predict ordinate given an abscissa from new data (if we are in 2D)

![linear_regression](https://github.com/JhonSanz/linear_regression_gradient_descent/blob/master/linear_regression.png?raw=true)



# Linear regression model

## Hypotesis hθ(X)

- 2-dimensional straight line:

  hθ(X) = θ0X0 + θ1X1

- In general:

  hθ(X) = θ0X0 + θ1X1 + ... + θnXn

X0 = 1, called bias

Written vectorized, simplifies our equation

- hθ(X) = θTX

where θ is our n-dimensional parameters vector

## Cost Function

Our cost function is going to be squared error function, that computes distance for every training examples. We need to minimice this function, because our job is to fit the better straight line that describes our data, so:

J(θ0, θ1) = 1/2m sum([(hθ(Xi) - yi)**2 for i in range(m)])


# Gradient Descent

To minimice using gradient descent we need derivate terms of our cost function, so, we use some calculus over our squared error function and it gave us this:

- θ = 1/2m sum([(hθ(Xi) - yi)**2 for i in range(m)])xj

We have derivated over θ

Written vectorized, simplifies our equation

- θ = 1/2m XT (hθ(Xi) - y)

And Gradient Descent algorithm is defined by

- θ = θ - α d/dθ J(θ)

Where α is the learning rate

So

- θ = θ - α (1/2m XT (hθ(Xi) - y))


Thanks to Coursera and Andrew Ng, I encourage you to take this course:
https://www.coursera.org/learn/machine-learning/home/welcome

Regards :)