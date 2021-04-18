import numpy as np
import matplotlib.pyplot as plt


def compute_u(theta_0, theta_1, xi, yi):
    return (theta_0 + (xi * theta_1)) - yi


def compute_cost(theta_0, theta_1, data):
    x = np.array(data[:, 46])
    x = np.delete(x, 0)
    x = np.array([minmax_scaling(xi, np.min(x), np.max(x)) for xi in x])
    y = np.array(data[:, 80])
    y = np.delete(y, 0)

    summation = 0
    for i in range(x.size):
        summation += pow(compute_u(theta_0, theta_1, x[i], y[i]), 2)

    total_cost = summation / x.size

    return total_cost


def step_gradient(theta_0_current, theta_1_current, data, alpha):
    x = np.array(data[:, 46])
    x = np.delete(x, 0)
    x = np.array([minmax_scaling(xi, np.min(x), np.max(x)) for xi in x])
    y = np.array(data[:, 80])
    y = np.delete(y, 0)

    summation0 = 0
    summation1 = 0
    for i in range(x.size):
        u = compute_u(theta_0_current, theta_1_current, x[i], y[i])
        summation0 += u * 1
        summation1 += u * x[i]

    theta_0_updated = theta_0_current - (2 * alpha) * (summation0 / x.size)
    theta_1_updated = theta_1_current - (2 * alpha) * (summation1 / x.size)

    return theta_0_updated, theta_1_updated


def gradient_descent(data, starting_theta_0, starting_theta_1, learning_rate, num_iterations):
    # valores iniciais
    theta_0 = starting_theta_0
    theta_1 = starting_theta_1

    # variável para armazenar o custo ao final de cada step_gradient
    cost_graph = []

    # vetores para armazenar os valores de Theta0 e Theta1 apos cada iteração de step_gradient (pred = Theta1*x + Theta0)
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


house_prices_data = np.genfromtxt('house_prices_train.csv', delimiter=',')

# Extrair colunas para análise
living_area = np.array(house_prices_data[:, 46])
living_area = np.delete(living_area, 0)
sales_price = np.array(house_prices_data[:, 80])
sales_price = np.delete(sales_price, 0)

living_area = np.array([minmax_scaling(x, np.min(living_area), np.max(living_area)) for x in living_area])

theta_0, theta_1, cost_graph, theta_0_progress, theta_1_progress = gradient_descent(house_prices_data,
                                                                                    starting_theta_0=0,
                                                                                    starting_theta_1=0,
                                                                                    learning_rate=0.01,
                                                                                    num_iterations=100)

#Imprimir parâmetros otimizados
print ('Theta_0 otimizado: ', theta_0)
print ('Theta_1 otimizado: ', theta_1)
print ('Custo minimizado: ', compute_cost(theta_0, theta_1, house_prices_data))

#Gráfico de dispersão do conjunto de dados
plt.figure(figsize=(10, 6))
plt.scatter(living_area, sales_price)
pred = theta_1 * living_area + theta_0
plt.plot(living_area, pred, c='r')
plt.xlabel('Área da Sala de Estar')
plt.ylabel('Preço')
plt.title('Data')
plt.show()
