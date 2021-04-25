import numpy as np
import matplotlib.pyplot as plt
import sys

# >>> DESCOMENTAR PARA ENTREGA <<<
# sys_argv = sys.argv
# file_argv = sys_argv[1]
# iterations_argv = sys_argv[2]

# >>> COMENTAR PARA ENTREGA <<<
file_argv = 'house_prices_train.csv'
iterations_argv = 300

def print_graph(data, is_theta=False, is_cost=False):
    living_area, overall_quality, overall_condition, garage_area, year_built, sales_price = get_properties(data)

    plt.figure(figsize=(10, 6))

    if is_theta:
        plt.scatter(range(iterations), theta0_progress)
        plt.scatter(range(iterations), theta1_progress)
        plt.scatter(range(iterations), theta2_progress)
        plt.scatter(range(iterations), theta3_progress)
        plt.scatter(range(iterations), theta4_progress)
        plt.scatter(range(iterations), theta5_progress)
        plt.xlabel('Thetas')
        plt.ylabel('Iterações')
    elif is_cost:
        plt.scatter(range(iterations), cost)
        plt.xlabel('Custo')
        plt.ylabel('Iterações')
    else:
        parameters = living_area + overall_quality + overall_condition + garage_area + year_built
        plt.scatter(parameters, sales_price)
        pred = theta0 + (living_area * theta1) + (overall_quality * theta2) + (overall_condition * theta3) + (garage_area * theta4) + (year_built * theta5)
        plt.scatter(parameters, pred)
        plt.xlabel('Parametros')
        plt.ylabel('Preço')
    plt.title('Data')
    plt.show()


def get_properties(data):
    e1 = np.array(data[:, 46])
    e1 = np.delete(e1, 0)
    e1 = np.array([minmax_scaling(xi, np.min(e1), np.max(e1)) for xi in e1])
    e2 = np.array(data[:, 17])
    e2 = np.delete(e2, 0)
    e2 = np.array([minmax_scaling(xi, np.min(e2), np.max(e2)) for xi in e2])
    e3 = np.array(data[:, 18])
    e3 = np.delete(e3, 0)
    e3 = np.array([minmax_scaling(xi, np.min(e3), np.max(e3)) for xi in e3])
    e4 = np.array(data[:, 62])
    e4 = np.delete(e4, 0)
    e4 = np.array([minmax_scaling(xi, np.min(e4), np.max(e4)) for xi in e4])
    e5 = np.array(data[:, 19])
    e5 = np.delete(e5, 0)
    e5 = np.array([minmax_scaling(xi, np.min(e5), np.max(e5)) for xi in e5])
    y = np.array(data[:, 80])
    y = np.delete(y, 0)

    return e1, e2, e3, e4, e5, y


def h0(theta_0, theta_1, theta_2, theta_3, theta_4, theta_5, e1, e2, e3, e4, e5):
    return theta_0 + (e1 * theta_1) + (e2 * theta_2) + (e3 * theta_3) + (e4 * theta_4) + (e5 * theta_5)


def compute_u(theta_0, theta_1, theta_2, theta_3, theta_4, theta_5, e1, e2, e3, e4, e5, yi):
    return h0(theta_0, theta_1, theta_2, theta_3, theta_4, theta_5, e1, e2, e3, e4, e5) - yi


def compute_cost(theta_0, theta_1, theta_2, theta_3, theta_4, theta_5, data):
    e1, e2, e3, e4, e5, y = get_properties(data)

    summation = 0
    for i in range(y.size):
        summation += pow(compute_u(theta_0, theta_1, theta_2, theta_3, theta_4, theta_5, e1[i], e2[i], e3[i], e4[i], e5[i], y[i]), 2)

    total_cost = summation / y.size

    return total_cost


def step_gradient(theta_0_current, theta_1_current, theta_2_current, theta_3_current, theta_4_current, theta_5_current, data, alpha):
    e1, e2, e3, e4, e5, y = get_properties(data)

    summation0 = 0
    summation1 = 0
    summation2 = 0
    summation3 = 0
    summation4 = 0
    summation5 = 0
    for i in range(y.size):
        u = compute_u(theta_0_current, theta_1_current, theta_2_current, theta_3_current, theta_4_current, theta_5_current, e1[i], e2[i], e3[i], e4[i], e5[i], y[i])
        summation0 += u * 1
        summation1 += u * e1[i]
        summation2 += u * e2[i]
        summation3 += u * e3[i]
        summation4 += u * e4[i]
        summation5 += u * e5[i]

    theta_0_updated = theta_0_current - (2 * alpha) * (summation0 / y.size)
    theta_1_updated = theta_1_current - (2 * alpha) * (summation1 / y.size)
    theta_2_updated = theta_2_current - (2 * alpha) * (summation2 / y.size)
    theta_3_updated = theta_3_current - (2 * alpha) * (summation3 / y.size)
    theta_4_updated = theta_4_current - (2 * alpha) * (summation4 / y.size)
    theta_5_updated = theta_5_current - (2 * alpha) * (summation5 / y.size)

    return theta_0_updated, theta_1_updated, theta_2_updated, theta_3_updated, theta_4_updated, theta_5_updated


def gradient_descent(data, starting_theta_0, starting_theta_1, starting_theta_2, starting_theta_3, starting_theta_4, starting_theta_5, learning_rate, num_iterations):
    # valores iniciais
    theta_0 = starting_theta_0
    theta_1 = starting_theta_1
    theta_2 = starting_theta_2
    theta_3 = starting_theta_3
    theta_4 = starting_theta_4
    theta_5 = starting_theta_5

    # variável para armazenar o custo ao final de cada step_gradient
    cost_graph = []

    # vetores para armazenar Theta0 e Theta1 apos cada iteração de step_gradient (pred = Theta1*x + Theta0)
    theta_0_progress = []
    theta_1_progress = []
    theta_2_progress = []
    theta_3_progress = []
    theta_4_progress = []
    theta_5_progress = []

    # Para cada iteração, obtem novos (Theta0,Theta1,Theta2) e calcula o custo (EQM)
    for i in range(num_iterations):
        cost_graph.append(compute_cost(theta_0, theta_1, theta_2, theta_3, theta_4, theta_5, data))
        theta_0, theta_1, theta_2, theta_3, theta_4, theta_5 = step_gradient(theta_0, theta_1, theta_2, theta_3, theta_4, theta_5, data, learning_rate)
        theta_0_progress.append(theta_0)
        theta_1_progress.append(theta_1)
        theta_2_progress.append(theta_2)
        theta_3_progress.append(theta_3)
        theta_4_progress.append(theta_4)
        theta_5_progress.append(theta_5)

    return [theta_0, theta_1, theta_2, theta_3, theta_4, theta_5, cost_graph, theta_0_progress, theta_1_progress, theta_2_progress, theta_3_progress, theta_4_progress, theta_5_progress]


def minmax_scaling(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin)


house_prices_data = np.genfromtxt(file_argv, delimiter=',')

iterations = iterations_argv

theta0, theta1, theta2, theta3, theta4, theta5, cost, theta0_progress, theta1_progress, theta2_progress, theta3_progress, theta4_progress, theta5_progress = gradient_descent(
    house_prices_data,
    starting_theta_0=0,
    starting_theta_1=0,
    starting_theta_2=0,
    starting_theta_3=0,
    starting_theta_4=0,
    starting_theta_5=0,
    learning_rate=0.01,
    num_iterations=iterations)

# Imprimir parâmetros otimizados
print('theta_0: ', theta0)
print('theta_1: ', theta1)
print('theta_2: ', theta2)
print('theta_3: ', theta3)
print('theta_4: ', theta4)
print('theta_5: ', theta5)
print('Erro quadratico medio: ', compute_cost(theta0, theta1, theta2, theta3, theta4, theta5, house_prices_data))

# print_graph(house_prices_data, False, True)
