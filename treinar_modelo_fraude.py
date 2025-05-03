import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Criando um dataframe com dados simulados
dados = {
'valor_transacao': [100, 200, 300, 400, 500],
'tempo_conta': [1, 2, 3, 4, 5],
'num_transacoes': [5, 10, 15, 20, 25],
'pais_origem': [0, 0, 1, 1, 1],
'fraude': [0, 0, 1, 1, 1]
}

df = pd.DataFrame(dados)

# Separando variáveis
X = df[['valor_transacao', 'tempo_conta', 'num_transacoes', 'pais_origem']]
y = df['fraude']

# Treinando o modelo
modelo = LogisticRegression()
modelo.fit(X, y)

# Salvando o modelo
with open('modelo_fraude.pkl', 'wb') as f:
  pickle.dump(modelo, f)

print("Modelo salvo com sucesso!")