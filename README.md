# 🚗 MVP - Sistema de Suporte à Decisão: Previsão de Preço de Veículos Usados

## 📖 Definição do Problema
O mercado de veículos usados possui alta variação de preços, influenciada por fatores como ano, marca, modelo, quilometragem, estado de conservação e localização.  
Prever corretamente o preço de venda ajuda concessionárias, lojistas e consumidores a tomarem decisões mais informadas.  

Este projeto tem como objetivo **criar um modelo preditivo para estimar o preço de venda de veículos usados**, utilizando dados públicos de um dataset do Kaggle.

---

## 🎯 Hipótese
- **Hipótese principal:** Veículos mais novos e com menor quilometragem tendem a ter preços de venda mais altos.  
- **Hipóteses secundárias:**  
  - A marca e o modelo exercem grande influência no preço.  
  - Variáveis como estado e cor do veículo têm menor impacto.

---

## 📊 Conjunto de Dados
O dataset utilizado foi **[Car Prices Dataset (Kaggle)]([https://www.kaggle.com/](https://www.kaggle.com/datasets/syedanwarafridi/vehicle-sales-data/data))**.  

- Total de registros: ~150.000 veículos  
- Principais variáveis:  
  - `year` → Ano de fabricação  
  - `make` → Marca  
  - `model` → Modelo  
  - `body` → Tipo de carroceria  
  - `transmission` → Transmissão  
  - `condition` → Condição do veículo  
  - `odometer` → Quilometragem  
  - `state` → Estado (EUA)  
  - `sellingprice` → Preço de venda (variável alvo)

---

## 🛠️ Metodologia

1. **Importação dos dados** via Kaggle API e Pandas.  
2. **EDA (Exploração dos Dados):**  
   - Identificação de valores nulos.  
   - Análise estatística das variáveis.  
   - Identificação de outliers (carros com preços ou quilometragens fora da faixa).  
3. **Pré-processamento:**  
   - Tratamento de valores faltantes.  
   - Criação da variável `vehicle_age` (idade do carro).  
   - Tratamento da variável `odometer` (eliminação de valores fora da faixa).  
   - Redução da cardinalidade (agrupamento de marcas e modelos raros).  
4. **Modelagem:**  
   - Modelos testados: **Baseline (Dummy Median), Regressão Linear, Random Forest**.  
   - Divisão do dataset em **80% treino, 10% validação e 10% teste**.   
   - Métricas utilizadas: MAE, RMSE e R².  
5. **Avaliação e Seleção:**  
   - Comparação entre os modelos.  
   - Escolha do **Random Forest** como melhor modelo.  
6. **Visualizações:**  
   - Distribuição de preços.  
   - Real vs. Predito.  
   - Distribuição dos resíduos.  
   - Distribuição dos anos.  
   - Importância das variáveis (Permutation Importance).  

---

## 📈 Resultados

### Modelos testados
- **Baseline (Dummy median)** → Serve apenas como referência mínima.  
- **Regressão Linear** → Melhor que o baseline, mas limitado em relações complexas.  
- **Random Forest** → Melhor desempenho geral.  

### Métricas finais do modelo escolhido (Random Forest):
- **MAE (Erro Médio Absoluto):** ≈ R$ 2.892  
- **RMSE (Raiz do Erro Quadrático Médio):** ≈ R$ 4.678  
- **R² (Coeficiente de Determinação):** ≈ 0.75  

### Interpretação:
- O modelo consegue explicar cerca de **75% da variação dos preços**.  
- Em média, o erro é de ≈ **R$ 2.900 por carro**.  
- As variáveis mais importantes foram:  
  - **Idade do veículo (`vehicle_age`)**  
  - **Quilometragem (`odometer`)**  
  - **Marca/Modelo (`make`, `model`)**

---

## 📊 Visualizações
- Distribuição dos preços → maioria dos carros custa menos de R$ 20.000, com outliers caros.  
- Real vs. Predito → bom ajuste no centro, maior erro nos extremos (carros muito baratos ou caros).  
- Resíduos → centrados em zero, mas com cauda longa (outliers).  
- Distribuição dos anos → maioria dos veículos entre 2007 e 2013.  
- Importância das features → idade, quilometragem e marca/modelo dominam.

---

## ✅ Conclusão
- A hipótese foi confirmada: carros mais novos e com menor quilometragem realmente tendem a ter preços mais altos.  
- O modelo **Random Forest** apresentou melhor desempenho entre os testados.  
- A análise mostrou coerência com o mercado automotivo real.  
-Isso confirma a hipótese: **carros mais novos e menos rodados valem mais**.

---

# Distrinchando o código

## 📌 Passo 1 – Importar bibliotecas

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')
```

📖 **Explicação:**

* `pandas` → manipulação de tabelas de dados.
* `numpy` → operações matemáticas e matrizes.
* `matplotlib.pyplot` → gráficos e visualização.
* `%matplotlib inline` → mostra os gráficos direto no notebook.
* `scikit-learn` → biblioteca de machine learning usada para pré-processar dados, criar modelos e avaliar métricas.
* `joblib` → salvar e carregar modelos treinados.
* `warnings` → controlar mensagens de aviso.

👉 **Por que?**
Essas bibliotecas são a base para análise, tratamento e modelagem dos dados.

---

## 📌 Passo 2 – Carregar a base de dados

```python
url = "https://raw.githubusercontent.com/ingredy3/SSD_1/refs/heads/main/car_prices.csv"
df = pd.read_csv(url)
print("Shape do dataset:", df.shape)
df.head()
```

📖 **Explicação:**

* `url` → link do arquivo CSV no GitHub.
* `pd.read_csv` → lê o arquivo e transforma em DataFrame.
* `df.shape` → mostra quantas linhas e colunas existem.
* `df.head()` → exibe as primeiras 5 linhas.

👉 **Por que?**
Carregar os dados é o primeiro passo para qualquer análise. Assim entendemos o tamanho e a estrutura da base.

---

## 📌 Passo 3 – EDA rápida (Exploração Inicial)

```python
df.info()
df.describe(include="all")
df.isnull().sum().sort_values(ascending=False).head(20)
```

📖 **Explicação:**

* `df.info()` → mostra tipos de dados (numérico, texto, etc.) e se há valores nulos.
* `df.describe()` → estatísticas básicas (média, mediana, mínimo, máximo).
* `df.isnull().sum()` → conta valores nulos por coluna.

👉 **Por que?**
Essa análise inicial permite identificar problemas de qualidade: dados faltando, variáveis numéricas/categóricas, presença de outliers.

---

## 📌 Passo 4 – Amostragem

```python
RANDOM_STATE = 42
df_sample = df.sample(frac=0.1, random_state=RANDOM_STATE)
df = df_sample.copy()
print("Shape amostra:", df.shape)
```

📖 **Explicação:**

* `.sample(frac=0.1)` → pega 10% da base.
* `random_state=42` → garante reprodutibilidade.
* `df = df_sample.copy()` → troca a base original pela amostra.

👉 **Por que?**
Deixar o código mais rápido para rodar sem perder representatividade dos dados.

---

## 📌 Passo 5 – Seleção de colunas

```python
candidate_cols = ["year", "make", "model", "body", "transmission",
                  "condition", "odometer", "color", "interior", "state"]
cols = [c for c in candidate_cols if c in df.columns]
target = "sellingprice"
df = df[cols + [target]].copy()
```

📖 **Explicação:**

* Lista apenas colunas úteis para prever preço.
* `target = "sellingprice"` → define variável que queremos prever.
* `df = df[...]` → mantém só as colunas escolhidas.

👉 **Por que?**
Reduzir ruído, mantendo apenas variáveis relevantes.

---

## 📌 Passo 6 – Criar variável `vehicle_age`

```python
df["vehicle_age"] = 2025 - df["year"]
df["vehicle_age"] = df["vehicle_age"].clip(lower=0, upper=100)
```

📖 **Explicação:**

* Calcula a idade do carro (2025 - ano).
* `.clip()` → limita para valores plausíveis (0 a 100 anos).

👉 **Por que?**
Idade do carro é mais informativa que o ano bruto.

---

## 📌 Passo 7 – Tratar `odometer`

```python
df["odometer"] = df["odometer"].where(
    (df["odometer"] >= 100) & (df["odometer"] <= 500000)
)
```

📖 **Explicação:**

* Mantém apenas valores entre 100 e 500.000 km.
* Valores fora disso viram `NaN`.

👉 **Por que?**
Evitar erros de digitação ou valores irreais (ex.: 999999 km).

---

## 📌 Passo 8 – Dataset final de modelagem

```python
df_model = df.copy()
df_model = df_model[df_model[target].notna()]
```

📖 **Explicação:**

* Copia a base já tratada.
* Remove linhas sem preço (`sellingprice`).

👉 **Por que?**
Um modelo não aprende sem o valor da variável alvo.

---

## 📌 Passo 9 – Reduzir cardinalidade

```python
def keep_top_n(series, top_n=80):
    top = series.value_counts().nlargest(top_n).index
    return series.where(series.isin(top), other="other")

for c in ["model","make"]:
    if c in df_model.columns:
        df_model[c] = keep_top_n(df_model[c], top_n=80)
```

📖 **Explicação:**

* Mantém apenas os 80 valores mais comuns de marca e modelo.
* Todos os outros viram “other”.

👉 **Por que?**
Reduzir categorias raras que não ajudam o modelo.

---

## 📌 Passo 10 – Divisão treino/validação/teste

```python
X = df_model.drop(columns=[target])
y = df_model[target]

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1111, random_state=42
)
```

📖 **Explicação:**

* Divide dados em 80% treino, 10% validação, 10% teste.
* `train` → usado para treinar.
* `val` → para escolher modelo.
* `test` → para avaliar resultado final.

👉 **Por que?**
Evita overfitting e garante avaliação justa.

---

## 📌 Passo 11 – Pré-processamento

```python
num_cols = [c for c in df_model.select_dtypes(include=["int64","float64"]).columns if c != target]
cat_cols = [c for c in df_model.select_dtypes(include=["object"]).columns]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])
```

📖 **Explicação:**

* Numéricas → substitui nulos pela mediana e padroniza escala.
* Categóricas → substitui nulos por “missing” e faz dummies (one-hot).

👉 **Por que?**
Modelos de ML só aceitam números e não funcionam bem com nulos.

---

## 📌 Passo 12 – Função de métricas

```python
def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "R2": r2}
```

📖 **Explicação:**

* MAE → erro médio em reais.
* RMSE → dá mais peso a erros grandes.
* R² → mede quanta variação do preço o modelo explica.

👉 **Por que?**
Precisamos de métricas para comparar modelos.

---

## 📌 Passo 13 – Modelo Baseline

```python
dummy = DummyRegressor(strategy="median")
pipe_dummy = Pipeline(steps=[("pre", preprocessor), ("model", dummy)])
pipe_dummy.fit(X_train, y_train)
y_val_pred_dummy = pipe_dummy.predict(X_val)
print("Baseline (validação):", regression_metrics(y_val, y_val_pred_dummy))
```

📖 **Explicação:**

* Modelo “burro” que sempre prevê a mediana.
* Serve como referência mínima.

👉 **Por que?**
Se o modelo inteligente não supera esse, ele não serve.

---

## 📌 Passo 14 – Regressão Linear

```python
lr = LinearRegression()
pipe_lr = Pipeline(steps=[("pre", preprocessor), ("model", lr)])
pipe_lr.fit(X_train, y_train)
y_val_pred_lr = pipe_lr.predict(X_val)
print("Linear Regression (validação):", regression_metrics(y_val, y_val_pred_lr))
```

📖 **Explicação:**

* Modelo que tenta ajustar uma linha reta entre variáveis e preço.

👉 **Por que?**
É rápido e simples, bom para benchmark.

---

## 📌 Passo 15 – Random Forest

```python
rf = RandomForestRegressor(
    n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
)
pipe_rf = Pipeline(steps=[("pre", preprocessor), ("model", rf)])
pipe_rf.fit(X_train, y_train)
y_val_pred_rf = pipe_rf.predict(X_val)
print("Random Forest (validação):", regression_metrics(y_val, y_val_pred_rf))
```

📖 **Explicação:**

* Cria várias árvores de decisão e combina os resultados.
* `n_estimators=100` → número de árvores.
* `max_depth=12` → profundidade máxima.

👉 **Por que?**
É mais flexível que a regressão linear e captura relações complexas.

---

## 📌 Passo 16 – Comparação geral

```python
print("Baseline:", regression_metrics(y_val, y_val_pred_dummy))
print("Linear Regression:", regression_metrics(y_val, y_val_pred_lr))
print("Random Forest:", regression_metrics(y_val, y_val_pred_rf))
```

📖 **Explicação:**

* Coloca as métricas lado a lado.

👉 **Por que?**
Permite escolher o melhor modelo de forma objetiva.

---

## 📌 Passo 17 – Modelo final

```python
X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])

final_model = pipe_rf.fit(X_trainval, y_trainval)

y_test_pred = final_model.predict(X_test)
print("Random Forest (teste final):", regression_metrics(y_test, y_test_pred))

joblib.dump(final_model, "model_final.joblib")
```

📖 **Explicação:**

* Junta treino+validação.
* Treina Random Forest no conjunto completo.
* Avalia no teste.
* Salva modelo em arquivo `.joblib`.

👉 **Por que?**
Treinar com mais dados aumenta performance. Avaliar no teste garante generalização.

---

## 📌 Passo 18 – Gráficos

```python
# Distribuição do preço
plt.hist(y, bins=50, range=(0,100000))
plt.title("Distribuição do Preço de Venda")
plt.show()

# Real x Predito
plt.scatter(y_test, y_test_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.title("Random Forest: Real vs Predito")
plt.show()

# Resíduos
residuos = y_test - y_test_pred
plt.hist(residuos, bins=50)
plt.title("Distribuição dos Resíduos")
plt.show()
```

📖 **Explicação:**

* Histogramas → mostram distribuição dos preços e dos erros.
* Scatter → compara valores reais com previstos.

👉 **Por que?**
Visualizações ajudam a entender se o modelo erra muito ou pouco, e em quais casos.

---

## 📌 Passo 19 – Importância das variáveis

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(final_model, X_test, y_test, n_repeats=5, random_state=42)

importances = pd.DataFrame({
    "feature": X_test.columns,
    "importance": result.importances_mean
}).sort_values(by="importance", ascending=False)

print(importances.head(15))
```

📖 **Explicação:**

* Mede quanto o modelo depende de cada variável.
* Mostra ranking das mais importantes.

👉 **Por que?**
Ajuda a interpretar o modelo e validar se ele aprendeu algo coerente com a realidade.

---


