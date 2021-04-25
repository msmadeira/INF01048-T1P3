import numpy as np
import matplotlib.pyplot as plt
import sys

sys_argv = sys.argv
file_argv = sys_argv[1]
iterations_argv = sys_argv[2]

// >>> COMENTAR PARA ENTREGA <<<
file_argv = 'house_prices_train.csv'
iterations_argv = 300

def print_graph(data, is_theta=False, is_cost=False):
    living_area, sales_price = get_properties(data)

    plt.figure(figsize=(10, 6))

    if is_theta:
        plt.scatter(range(iterations), theta0_progress)
        plt.scatter(range(iterations), theta1_progress)
        plt.xlabel('Thetas')
        plt.ylabel('Iterações')
    elif is_cost:
        plt.scatter(range(iterations), cost)
        plt.xlabel('Custo')
        plt.ylabel('Iterações')
    else:
        plt.scatter(living_area, sales_price)
        pred = x = np.array([h0(theta0, theta1, xi) for xi in living_area])
        plt.scatter(living_area, pred)
        plt.xlabel('Área da Sala de Estar')
        plt.ylabel('Preço')
    plt.title('Data')
    plt.show()


def get_properties(data):
    x = np.array(data[:, 46])
    x = np.delete(x, 0)
    x = np.array([minmax_scaling(xi, np.min(x), np.max(x)) for xi in x])
    y = np.array(data[:, 80])
    y = np.delete(y, 0)

    return x, y


def h0(theta_0, theta_1, xi):
    return theta_0 + (xi * theta_1)


def compute_u(theta_0, theta_1, xi, yi):
    return h0(theta_0, theta_1, xi) - yi


def compute_cost(theta_0, theta_1, data):
    x, y = get_properties(data)

    summation = 0
    for i in range(x.size):
        summation += pow(compute_u(theta_0, theta_1, x[i], y[i]), 2)

    total_cost = summation / x.size

    return total_cost


def step_gradient(theta_0_current, theta_1_current, data, alpha):
    x, y = get_properties(data)

    summation0 = 0
    summation1 = 0
    for i in range(x.size):
        u = compute_u(theta_0_current, theta_1_current, x[i], y[i])
        summation0 += (u * 1)
        summation1 += (u * x[i])

    theta_0_updated = theta_0_current - ((2 * alpha) * (summation0 / x.size))
    theta_1_updated = theta_1_current - ((2 * alpha) * (summation1 / x.size))

    return theta_0_updated, theta_1_updated


def gradient_descent(data, starting_theta_0, starting_theta_1, learning_rate, num_iterations):
    # valores iniciais
    theta_0 = starting_theta_0
    theta_1 = starting_theta_1

    # variável para armazenar o custo ao final de cada step_gradient
    cost_graph = []

    # vetores para armazenar Theta0 e Theta1 apos cada iteração de step_gradient (pred = Theta1*x + Theta0)
    theta_0_progress = []
    theta_1_progress = []

    # Para cada iteração, obtem novos (Theta0,Theta1) e calcula o custo (EQM)
    for i in range(num_iterations):
        cost_graph.append(compute_cost(theta_0, theta_1, data))
        theta_0, theta_1 = step_gradient(theta_0, theta_1, data, learning_rate)
        theta_0_progress.append(theta_0)
        theta_1_progress.append(theta_1)

    return [theta_0, theta_1, cost_graph, theta_0_progress, theta_1_progress]


def minmax_scaling(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin)


house_prices_data = np.genfromtxt(file_argv, delimiter=',')

iterations = iterations_argv

theta0, theta1, cost, theta0_progress, theta1_progress = gradient_descent(house_prices_data,
                                                                          starting_theta_0=0,
                                                                          starting_theta_1=0,
                                                                          learning_rate=0.01,
                                                                          num_iterations=iterations)

# Imprimir parâmetros otimizados
print('theta_0: ', theta0)
print('theta_1: ', theta1)
print('Erro quadratico medio: ', compute_cost(theta0, theta1, house_prices_data))

# print_graph(house_prices_data, False, True)
