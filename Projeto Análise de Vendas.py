


# Manipulação de dados
import pandas as pd

# Divide os dados em treino e teste
from sklearn.model_selection import train_test_split

# Converte textos em números
from sklearn.preprocessing import OneHotEncoder

# Aplica transformações nas colunas
from sklearn.compose import ColumnTransformer

# Modelo de Regressão Linear
from sklearn.linear_model import LinearRegression

# Métricas de avaliação
from sklearn.metrics import mean_absolute_error, r2_score

#Carregando o dataset

df = pd.read_csv("/content/Vendasloja - Vendasloja.csv")

#carregar as primeiras linhas

df.head()

#Quantidade de linhas e colunas
df.shape

#Informações do dataset
df.info()

#Estatísticas descritivas
df.describe()

#identificar valores ausentes

df.isnull().sum()

"""Tratar os valores ausentes"""

#Substituindo os valores numéricos pela mediana.
df["Qtd"] = df["Qtd"].fillna(df["Qtd"].median())

df["Lucro"] = df["Lucro"].fillna(df["Lucro"].median())

#Substituindo a coluna categórica pela moda.
df["Cidade"] = df["Cidade"].fillna(
df["Cidade"].mode()[0]
)

#Conferindo novamente.
df.isnull().sum()

#Separando X e y
# Variável alvo
y = df["Lucro"]

# Variáveis de entrada
X = df.drop(columns=["ID","Lucro"])

#Identificando colunas categóricas
categoricas = [
    "Produto",
    "Categoria",
    "Cidade",
    "ClienteVIP",
    "FormaPagamento",
    "DiaSemana"]

#Criando o One Hot Encoder
#trasnforma as variáveis categoricas em numéricas
preprocessador = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categoricas
        )
    ],
    remainder="passthrough"
)

#Aplicando a transformação

X = preprocessador.fit_transform(X)

#Dividindo treino e teste
X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.20,
random_state=42
)

#Criando o modelo
modelo = LinearRegression()

#Treinando o modelo
modelo.fit(X_train, y_train)

#Fazendo previsões
y_pred = modelo.predict(X_test)

#Avaliando o modelo
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")

#Comparando valores reais e previstos
resultado = pd.DataFrame({
    "Lucro Real": y_test,
    "Lucro Previsto": y_pred
})

resultado.head(10)

"""Parte 2 – Comparando Modelos: Árvore de Decisão
Na primeira parte da aula utilizamos a Regressão Linear para prever o lucro
das vendas. Agora vamos responder à mesma pergunta de negócio utilizando
outro algoritmo: Árvore de Decisão. Ao final, compararemos os resultados dos
dois modelos utilizando as métricas MAE e R².
"""

#   Criar o modelo de Árvore de Decisão
from sklearn.tree import DecisionTreeRegressor
modelo_arvore = DecisionTreeRegressor(random_state=42)

#Treinando o modelo
modelo_arvore.fit(
X_train,
y_train
)

#Fazendo previsões

y_pred_arvore = modelo_arvore.predict(
X_test
)

#Avaliando o modelo

mae_arvore = mean_absolute_error(
y_test,
y_pred_arvore
)

r2_arvore = r2_score(
y_test,
y_pred_arvore
)

print(f"MAE: {mae_arvore:.2f}")
print(f"R²: {r2_arvore:.2f}")

#Comparando valores reais e previstos
resultado_arvore = pd.DataFrame({
"Lucro Real": y_test,
"Lucro Previsto": y_pred_arvore
})

resultado_arvore.head(10)

#Comparação dos modelos

#Agora juntamos os resultados da Regressão Linear e da Árvore de
comparacao = pd.DataFrame({
    "Modelo": ["Regressão Linear", "Árvore de Decisão"],
    "MAE": [mae, mae_arvore],
    "R²": [r2, r2_arvore]
})

comparacao

import matplotlib.pyplot as plt

# Gráfico: valores reais x valores previstos

plt.figure(figsize=(10,5))

# Regressão Linear
plt.scatter(y_test, y_pred,
            label="Regressão Linear", alpha=0.7)

# Árvore de Decisão
plt.scatter(y_test, y_pred_arvore,
            label="Árvore de Decisão", alpha=0.7)

# Linha ideal (previsão perfeita)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linestyle="--",
         label="Previsão perfeita")

plt.xlabel("Valor Real")
plt.ylabel("Valor Previsto")

plt.title("Comparação: Valores Reais x Valores Previstos")
plt.legend()
plt.grid(True)

plt.show()

"""Conclusão

Este projeto teve como objetivo desenvolver e comparar modelos de aprendizado de máquina capazes de prever o lucro de uma empresa a partir de diferentes características presentes no conjunto de dados. Para isso, foi seguido um fluxo completo de desenvolvimento de um modelo preditivo, desde o tratamento dos dados até a avaliação dos resultados.

Na primeira etapa, foi realizada a separação das variáveis independentes (X) e da variável alvo (Lucro), permitindo que os algoritmos identificassem corretamente quais informações seriam utilizadas para realizar as previsões. Em seguida, as variáveis categóricas, como produto, categoria, cidade, cliente VIP, forma de pagamento e dia da semana, foram transformadas em variáveis numéricas por meio da técnica One-Hot Encoding, utilizando o ColumnTransformer. Esse processo foi essencial para que os algoritmos de regressão pudessem interpretar corretamente essas informações durante o treinamento.

Após o pré-processamento, o conjunto de dados foi dividido em dois grupos: 80% para treinamento e 20% para teste. Essa divisão permitiu treinar os modelos utilizando a maior parte dos dados e avaliar seu desempenho em dados nunca vistos, reduzindo o risco de superestimação dos resultados.

Na etapa de modelagem, foram treinados dois algoritmos de regressão: a Regressão Linear e a Árvore de Decisão para Regressão. Ambos os modelos foram avaliados por meio das métricas MAE (Erro Absoluto Médio), que mede o erro médio das previsões, e R² (Coeficiente de Determinação), que indica o quanto o modelo consegue explicar a variação dos dados.

Os resultados mostraram que a Regressão Linear obteve um MAE de 390,89 e um R² de 0,82, apresentando um bom desempenho na previsão do lucro. Entretanto, o modelo de Árvore de Decisão apresentou resultados superiores, alcançando um MAE de 190,18 e um R² de 0,87, o que significa que suas previsões ficaram mais próximas dos valores reais e que o modelo conseguiu explicar uma parcela maior da variação do lucro.

Por fim, foi construída uma visualização gráfica comparando os valores reais e previstos pelos dois modelos. Essa análise visual confirmou o melhor desempenho da Árvore de Decisão, demonstrando que suas previsões se aproximaram mais da linha de previsão ideal.

Dessa forma, conclui-se que todas as etapas do projeto — desde o pré-processamento dos dados, codificação das variáveis categóricas, divisão dos dados, treinamento dos modelos, avaliação por métricas e análise gráfica — foram fundamentais para a construção de um modelo preditivo confiável. Entre os modelos testados, a Árvore de Decisão apresentou o melhor desempenho, tornando-se a opção mais adequada para estimar o lucro nesse conjunto de dados. Além disso, o projeto demonstrou a importância da preparação correta dos dados e da comparação entre diferentes algoritmos para identificar a solução mais eficiente para problemas de previsão, contribuindo para uma tomada de decisão mais precisa e baseada em dados.
"""
