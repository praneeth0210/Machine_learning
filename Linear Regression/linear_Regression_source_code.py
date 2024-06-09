import numpy as np
import matplotlib.pyplot as plt

def compute_error_for_line_given_points(c, m, points):
    total_error = np.sum((points[:, 1] - (m * points[:, 0] + c)) ** 2)
    return total_error

def gradient_descent_runner(points, initial_c, initial_m, learning_rate, num_iterations):
    m = initial_m
    c = initial_c
    for _ in range(num_iterations):
        m, c = step_gradient(c, m, points, learning_rate)
    return c, m

def step_gradient(c_current, m_current, points, learning_rate):
    N = float(len(points))
    x = points[:, 0]
    y = points[:, 1]
    
    c_gradient = -2 * np.sum(y - (m_current * x + c_current)) / N
    m_gradient = -2 * np.sum(x * (y - (m_current * x + c_current))) / N
    
    new_c = c_current - (learning_rate * c_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return new_c, new_m

def plot_points_and_lines(points, initial_c, initial_m, final_c, final_m):
    x = points[:, 0]
    y = points[:, 1]

    plt.scatter(x, y, color='blue', label='Data points',alpha=0.7)

    initial_y_pred = initial_m * x + initial_c
    plt.plot(x, initial_y_pred, color='red', label='Initial line')

    final_y_pred = final_m * x + final_c
    plt.plot(x, final_y_pred, color='green', label='Final line')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Gradient Descent Linear Regression')
    plt.show()

def run(file_path='data.csv'):
    try:
        points = np.genfromtxt(file_path, delimiter=',')
    except Exception as e:
        print(f"Error reading the data file: {e}")
        return
    
    # Define our hyperparameters
    learning_rate = 0.00009
    initial_c = 0
    initial_m = 0
    num_iterations = 1000

    # Train our model
    print(f'Starting gradient descent at c = {initial_c}, m = {initial_m}, error = {compute_error_for_line_given_points(initial_c, initial_m, points)}')
    c, m = gradient_descent_runner(points, initial_c, initial_m, learning_rate, num_iterations)
    print(f'Ending gradient descent at c = {c}, m = {m}, error = {compute_error_for_line_given_points(c, m, points)}')

    # Plot the results
    plot_points_and_lines(points, initial_c, initial_m, c, m)

# Run the gradient descent with the default file path
run()
