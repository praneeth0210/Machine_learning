import numpy as np
import pandas as pd

def compute_error_for_line_given_points(current_c, current_m, points):
    total_error = 0
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += (y - (current_m * x + current_c)) ** 2
    return total_error

def gradient_descent_runner(points, initial_c, initial_m, learning_rate, num_iteration):
    m = initial_m
    c = initial_c
    for i in range(num_iteration):
        m, c = step_gradient(c, m, points, learning_rate)
    return c, m

def step_gradient(c_current, m_current, points, learningRate):
    c_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        c_gradient += -(2/N) * (y - (m_current * x + c_current))
        m_gradient += (2/N) * x * (y - (m_current * x + c_current))

    new_c = c_current - (learningRate * c_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return new_c, new_m

def run():
    # Collect our data
    points = np.genfromtxt('python/.ipynb_checkpoints/data.csv', delimiter=',')

    # Define our hyperparameters
    learning_rate = 0.00009
    initial_c = 0
    initial_m = 0
    num_iteration = 1000

    # Train our model
    print(f'Starting gradient descent at b = {initial_c}, m = {initial_m}, error = {compute_error_for_line_given_points(initial_c, initial_m, points)}')
    c, m = gradient_descent_runner(points, initial_c, initial_m, learning_rate, num_iteration)
    print(f'Ending gradient descent at b = {c}, m = {m}, error = {compute_error_for_line_given_points(c, m, points)}')

run()
