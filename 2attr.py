import numpy as np


def compute_u(theta_0, theta_1, theta_2, xi, yi, zi):
    return (theta_0 + (xi * theta_1) + (zi * theta_2)) - yi


def compute_cost(theta_0, theta_1, theta_2, data):
    x = np.array(data[:, 46])
    x = np.delete(x, 0)
    x = np.array([minmax_scaling(xi, np.min(x), np.max(x)) for xi in x])
    y = np.array(data[:, 80])
    y = np.delete(y, 0)
    z = np.array(data[:, 17])
    z = np.delete(z, 0)

    summation = 0
    for i in range(x.size):
        summation += pow(compute_u(theta_0, theta_1, theta_2, x[i], y[i], z[i]), 2)

    total_cost = summation / x.size

    return total_cost


def step_gradient(theta_0_current, theta_1_current, theta_2_current, data, alpha):
    x = np.array(data[:, 46])
    x = np.delete(x, 0)
    x = np.array([minmax_scaling(xi, np.min(x), np.max(x)) for xi in x])
    y = np.array(data[:, 80])
    y = np.delete(y, 0)
    z = np.array(data[:, 17])
    z = np.delete(z, 0)

    summation0 = 0
    summation1 = 0
    summation2 = 0
    for i in range(x.size):
        u = compute_u(theta_0_current, theta_1_current, theta_2_current, x[i], y[i], z[i])
        summation0 += u * 1
        summation1 += u * x[i]
        summation2 += u * z[i]

    theta_0_updated = theta_0_current - (2 * alpha) * (summation0 / x.size)
    theta_1_updated = theta_1_current - (2 * alpha) * (summation1 / x.size)
    theta_2_updated = theta_2_current - (2 * alpha) * (summation2 / x.size)

    return theta_0_updated, theta_1_updated, theta_2_updated


def gradient_descent(data, starting_theta_0, starting_theta_1, starting_theta_2, learning_rate, num_iterations):
    # valores iniciais
    theta_0 = starting_theta_0
    theta_1 = starting_theta_1
    theta_2 = starting_theta_2

    # variável para armazenar o custo ao final de cada step_gradient
    cost_graph = []

    # vetores para armazenar os valores de Theta0 e Theta1 apos cada iteração de step_gradient (pred = Theta1*x + Theta0)
    theta_0_progress = []
    theta_1_progress = []
    theta_2_progress = []

    # Para cada iteração, obtem novos (Theta0,Theta1,Theta2) e calcula o custo (EQM)
    for i in range(num_iterations):
        cost_graph.append(compute_cost(theta_0, theta_1, theta_2, data))
        theta_0, theta_1, theta_2 = step_gradient(theta_0, theta_1, theta_2, data, learning_rate)
        theta_0_progress.append(theta_0)
        theta_1_progress.append(theta_1)
        theta_2_progress.append(theta_2)

    return [theta_0, theta_1, theta_2, cost_graph, theta_0_progress, theta_1_progress, theta_2_progress]


def minmax_scaling(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin)


house_prices_data = np.genfromtxt('house_prices_train.csv', delimiter=',')

theta_0, theta_1, theta_2, cost_graph, theta_0_progress, theta_1_progress, theta_2_progress = gradient_descent(
    house_prices_data,
    starting_theta_0=0,
    starting_theta_1=0,
    starting_theta_2=0,
    learning_rate=0.1,
    num_iterations=100)

# Imprimir parâmetros otimizados
print('theta_0: ', theta_0)
print('theta_1: ', theta_1)
print('theta_2: ', theta_2)
print('Erro quadratico medio: ', compute_cost(theta_0, theta_1, theta_2, house_prices_data))