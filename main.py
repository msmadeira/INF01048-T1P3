import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('house_prices_train.csv', delimiter=',')

#Extrair colunas para análise
ids = np.array(data[:, 0])
living_area = np.array(data[:, 46])

#Gráfico dos dados
plt.figure(figsize=(20, 12))
plt.scatter(ids, living_area)
plt.xlabel('Ids')
plt.ylabel('Área da Sala de Estar')
plt.title('Dados')
plt.show()
