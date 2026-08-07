 📊 Projeto de Análise de Vendas e Previsão de Lucro

 Sobre o Projeto

Este projeto apresenta uma aplicação prática de **Ciência de Dados e Machine Learning** para análise de dados de vendas e **previsão de lucro**.

O objetivo foi utilizar dados de vendas para realizar uma preparação estruturada dos dados, aplicar técnicas de pré-processamento e desenvolver modelos de **Regressão Linear** e **Árvore de Decisão**, comparando o desempenho dos algoritmos por meio das métricas **MAE** e **R²**.

O projeto foi desenvolvido em **Python**, utilizando principalmente a biblioteca **Pandas** para manipulação dos dados e **Scikit-learn** para construção, treinamento e avaliação dos modelos.


 Objetivos

* Realizar análise inicial de um conjunto de dados de vendas;
* Identificar e tratar valores ausentes;
* Preparar variáveis categóricas para utilização em Machine Learning;
* Separar variáveis preditoras e variável-alvo;
* Dividir os dados entre treinamento e teste;
* Desenvolver um modelo de **Regressão Linear**;
* Desenvolver um modelo de **Árvore de Decisão**;
* Realizar previsões de lucro;
* Avaliar os modelos utilizando **MAE** e **R²**;
* Comparar os resultados e identificar o modelo com melhor desempenho.


 Dataset

O conjunto de dados possui **250 registros e 12 variáveis**, relacionadas a vendas de produtos.

#Principais variáveis

| Variável         | Descrição                    |
| ---------------- | ---------------------------- |
| `ID`             | Identificador da venda       |
| `Produto`        | Produto comercializado       |
| `Categoria`      | Categoria do produto         |
| `Qtd`            | Quantidade vendida           |
| `Preco`          | Preço do produto             |
| `Desconto`       | Percentual de desconto       |
| `Cidade`         | Cidade da venda              |
| `ClienteVIP`     | Identificação de cliente VIP |
| `Frete`          | Valor do frete               |
| `FormaPagamento` | Forma de pagamento           |
| `DiaSemana`      | Dia da semana da venda       |
| `Lucro`          | Variável utilizada como alvo |

O dataset apresentou valores ausentes nas colunas `Qtd`, `Cidade` e `Lucro`, com 10 registros ausentes em cada uma dessas variáveis.

---

## 🔎 Análise e Preparação dos Dados

Inicialmente foram realizadas análises exploratórias utilizando recursos do **Pandas**, incluindo:

* Visualização das primeiras linhas;
* Verificação da quantidade de linhas e colunas;
* Análise dos tipos de dados;
* Estatísticas descritivas;
* Identificação de valores ausentes.

O dataset apresentou dimensões de **250 linhas × 12 colunas**.

### Tratamento de valores ausentes

Para as variáveis numéricas:

* `Qtd` → valores ausentes substituídos pela **mediana**;
* `Lucro` → valores ausentes substituídos pela **mediana**.

Para a variável categórica:

* `Cidade` → valores ausentes substituídos pela **moda**.

Após o tratamento, não permaneceram valores ausentes no conjunto de dados.

---

## ⚙️ Preparação para Machine Learning

A variável **`Lucro`** foi definida como variável-alvo:

```python
y = df["Lucro"]
```

As demais variáveis utilizadas como entrada foram separadas em `X`, removendo as colunas `ID` e `Lucro`.

As variáveis categóricas foram identificadas e transformadas utilizando **One-Hot Encoding**, por meio do `OneHotEncoder` e `ColumnTransformer`.

O conjunto de dados foi posteriormente dividido em:

80% para treinamento**
  20% para teste**

utilizando `train_test_split` com `random_state=42`.


## 🤖 Modelos de Machine Learning

1. Regressão Linear

O primeiro modelo utilizado foi a **Regressão Linear**, aplicada para estimar o lucro das vendas.

```python
modelo = LinearRegression()
modelo.fit(X_train, y_train)


Resultado obtido:

| Métrica |  Resultado |
| ------- | ---------: |
| MAE     | **390,89** |
| R²      |   **0,82** |

O modelo apresentou um **R² de aproximadamente 0,82**, indicando que conseguiu explicar uma parcela relevante da variação observada no lucro.

2. Árvore de Decisão

Na segunda etapa foi desenvolvido um modelo de **Árvore de Decisão para regressão** utilizando `DecisionTreeRegressor`.

python
modelo_arvore = DecisionTreeRegressor(random_state=42)
modelo_arvore.fit(X_train, y_train)


Resultado obtido:

| Métrica |  Resultado |
| ------- | ---------: |
| MAE     | **190,18** |
| R²      |   **0,87** |

A Árvore de Decisão apresentou desempenho superior ao modelo de Regressão Linear nas duas métricas utilizadas.


 Comparação dos Modelos

| Modelo                |        MAE |       R² |
| --------------------- | ---------: | -------: |
| Regressão Linear      |     390,89 |     0,82 |
| **Árvore de Decisão** | **190,18** | **0,87** |

 Melhor modelo

Com base nos resultados obtidos no projeto, a **Árvore de Decisão** apresentou o melhor desempenho.

Ela apresentou:
Menor MAE:** 190,18 contra 390,89;
  Maior R²:** 0,87 contra 0,82.

Portanto, dentro da avaliação realizada neste notebook, a **Árvore de Decisão foi o modelo que apresentou melhor capacidade de previsão do lucro**.

Visualização dos Resultados

O projeto também realizou uma comparação visual entre:

* Valores reais de lucro;
* Valores previstos pela Regressão Linear;
* Valores previstos pela Árvore de Decisão;
* Linha de referência correspondente à previsão perfeita.

Essa visualização permite observar a proximidade entre os valores reais e as previsões produzidas pelos modelos.


Tecnologias e Bibliotecas

   Linguagem
    Python
 
 Manipulação e análise de dados
  Pandas

Machine Learning
* **Scikit-learn**
* `LinearRegression`
* `DecisionTreeRegressor`
* `train_test_split`
* `OneHotEncoder`
* `ColumnTransformer`

 Métricas
* `mean_absolute_error`
* `r2_score`

Visualização

Matplotlib

Pipeline do Projeto

Dataset de Vendas
        ↓
Carregamento dos Dados
        ↓
Análise Exploratória
        ↓
Identificação de Valores Ausentes
        ↓
Tratamento dos Dados
        ↓
Separação de X e y
        ↓
Codificação das Variáveis Categóricas
        ↓
Divisão Treino/Teste
        ↓
┌───────────────────────┐
│                       │
▼                       ▼
Regressão Linear    Árvore de Decisão
│                       │
▼                       ▼
Previsões             Previsões
│                       │
└───────────┬───────────┘
            ↓
     Avaliação dos Modelos
            ↓
       MAE e R²
            ↓
      Comparação Final
```

---

Principais Aprendizados

Este projeto permitiu aplicar, de forma prática, conceitos importantes de **Ciência de Dados e Machine Learning**, incluindo:

* Manipulação de datasets com Pandas;
* Análise exploratória de dados;
* Identificação e tratamento de dados ausentes;
* Engenharia de atributos categóricos;
* One-Hot Encoding;
* Separação entre variáveis de entrada e variável-alvo;
* Divisão de dados em treino e teste;
* Treinamento de modelos de Machine Learning;
* Regressão supervisionada;
* Avaliação de modelos;
* Comparação de algoritmos;
* Interpretação de métricas de desempenho;
* Visualização de previsões.

---

 Estrutura do Projeto


Projeto_Vendas/
│
├── Projeto_Vendas.ipynb
├── Vendasloja - Vendasloja.csv
└── README.md
```

> **Observação:** o arquivo CSV deve estar disponível no mesmo ambiente do notebook ou ter seu caminho ajustado no código de carregamento dos dados.

---

## ▶️ Como Executar

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/Projeto_Vendas.git
```

### 2. Acesse a pasta

```bash
cd Projeto_Vendas
```

### 3. Instale as dependências

```bash
pip install pandas scikit-learn matplotlib
```

### 4. Execute o notebook

O projeto pode ser executado utilizando:

* Jupyter Notebook;
* JupyterLab;
* Google Colab;
* VS Code com suporte a notebooks.

---

## 📌 Conclusão

O projeto demonstra um fluxo completo de aplicação de **Machine Learning em um problema de previsão de lucro**, desde a preparação dos dados até a avaliação e comparação dos modelos.

Entre os algoritmos avaliados, a **Árvore de Decisão apresentou o melhor resultado**, alcançando **MAE de 190,18** e **R² de 0,87**, enquanto a Regressão Linear apresentou **MAE de 390,89** e **R² de 0,82**.

Além de demonstrar conhecimentos técnicos em Python e Machine Learning, o projeto evidencia a capacidade de transformar dados de vendas em informações que podem apoiar análises e decisões orientadas por dados.

---

 Autor

Renan Henrique Martins Cunha

Estudante e profissional em formação nas áreas de:

* Ciência de Dados
* Inteligência Artificial
* Machine Learning
* Python
* Análise de Dados
  

📌 Projeto desenvolvido para fins de estudo e portfólio profissional em Ciência de Dados e Machine Learning.
