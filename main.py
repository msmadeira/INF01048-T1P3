import numpy as np
import matplotlib.pyplot as plt


def h_theta(x, theta_0, theta_1):
    return theta_0 + (x * theta_1)


def compute_cost(theta_0, theta_1, data):
    x = np.array(data[:, 46])
    x = np.delete(x, 0)
    x = np.array([minmax_scaling(xi, np.min(x), np.max(x)) for xi in x])
    y = np.array(data[:, 80])
    y = np.delete(y, 0)

    summation = 0
    for i in range(x.size):
        u = (theta_0 + (x[i] * theta_1)) - y[i]
        summation += pow(u, 2)

    total_cost = summation / x.size

    return total_cost


def step_gradient(theta_0_current, theta_1_current, data, alpha):
    """Calcula um passo em direção ao EQM mínimo

    Args:
        theta_0_current (float): valor atual de theta_0
        theta_1_current (float): valor atual de theta_1
        data (np.array): vetor com dados de treinamento (x,y)
        alpha (float): taxa de aprendizado / tamanho do passo

    Retorna:
        tupla: (theta_0, theta_1) os novos valores de theta_0, theta_1
    """

    theta_0_updated = 0
    theta_1_updated = 0

    ### SEU CODIGO AQUI

    return theta_0_updated, theta_1_updated


def gradient_descent(data, starting_theta_0, starting_theta_1, learning_rate, num_iterations):
    """executa a descida do gradiente

    Args:
        data (np.array): dados de treinamento, x na coluna 0 e y na coluna 1
        starting_theta_0 (float): valor inicial de theta0
        starting_theta_1 (float): valor inicial de theta1
        learning_rate (float): hyperparâmetro para ajustar o tamanho do passo durante a descida do gradiente
        num_iterations (int): hyperparâmetro que decide o número de iterações que cada descida de gradiente irá executar

    Retorna:
        list : os primeiros dois parâmetros são o Theta0 e Theta1, que armazena o melhor ajuste da curva. O terceiro e quarto parâmetro, são vetores com o histórico dos valores para Theta0 e Theta1.
    """

    # valores iniciais
    theta_0 = starting_theta_0
    theta_1 = starting_theta_1

    # variável para armazenar o custo ao final de cada step_gradient
    cost_graph = []

    # vetores para armazenar os valores de Theta0 e Theta1 apos cada iteração de step_gradient (pred = Theta1*x + Theta0)
    theta_0_progress = []
    theta_1_progress = []

    # Para cada iteração, obtem novos (Theta0,Theta1) e calcula o custo (EQM)
    num_iterations = 10
    for i in range(num_iterations):
        cost_graph.append(compute_cost(theta_0, theta_1, data))
        theta_0, theta_1 = step_gradient(theta_0, theta_1, data, alpha=0.0001)
        theta_0_progress.append(theta_0)
        theta_1_progress.append(theta_1)

    return [theta_0, theta_1, cost_graph, theta_0_progress, theta_1_progress]


def minmax_scaling(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin)


house_prices_data = np.genfromtxt('house_prices_train.csv', delimiter=',')

# Extrair colunas para análise
# ids = np.array(house_prices_data[:, 0])
# ids = np.delete(ids, 0)
# living_area = np.array(house_prices_data[:, 46])
# living_area = np.delete(living_area, 0)
# sales_price = np.array(house_prices_data[:, 80])
# sales_price = np.delete(sales_price, 0)
#
# living_area = np.array([minmax_scaling(x, np.min(living_area), np.max(living_area)) for x in living_area])
#
# plt.figure(figsize=(20, 12))
# plt.scatter(ids, living_area)
# plt.xlabel('Ids')
# plt.ylabel('Área da Sala de Estar')
# plt.title('Dados')
# plt.show()
