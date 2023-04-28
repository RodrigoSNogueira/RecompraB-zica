# importar as bibliotecas necessárias
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# carregar os dados
dados = pd.read_csv("recomprabazica.csv")

# selecionar os recursos relevantes
recursos = ['ID_Cliente']
recursos2 = ['Data', 'Bairro', 'Cidade']

# dividir os dados em recursos e rótulos
X = dados[recursos]
y = dados[recursos2]

# dividir os dados em conjuntos de treinamento e teste
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

# criar o modelo
modelo = DecisionTreeClassifier()

# treinar o modelo
modelo.fit(X_treinamento, y_treinamento)

# fazer previsões no conjunto de teste
previsoes = modelo.predict(X_teste)

# avaliar a precisão do modelo
precisao = accuracy_score(y_teste, previsoes)
print("Precisão:", precisao)
