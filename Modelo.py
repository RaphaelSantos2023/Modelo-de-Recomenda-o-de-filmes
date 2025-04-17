import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix

# 1. Carregar os dados
print(">>> Carregando os dados...")
baseRatings = pd.read_csv('ratings.csv', nrows=1000000)
baseMovies = pd.read_csv('movies.csv')

# 2. Preparar os dados para o Surprise
print(">>> Preparando os dados para o modelo...")
reader = Reader(rating_scale=(baseRatings['rating'].min(), baseRatings['rating'].max()))
data = Dataset.load_from_df(baseRatings[['userId', 'movieId', 'rating']], reader)

# 3. Dividir em treino e teste
print(">>> Dividindo os dados em treino e teste...")
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# 4. Treinar modelo SVD
print(">>> Treinando modelo SVD...")
model = SVD(random_state=42)
model.fit(trainset)

# 5. Previsões no conjunto de teste
print(">>> Realizando previsões...")
predictions = model.test(testset)

# 6. Avaliação do modelo com RMSE e MAE
print("\n>>> Avaliação do Modelo SVD (Regressão):")
rmse_val = accuracy.rmse(predictions)
mae_val = accuracy.mae(predictions)

# 7. Validação Cruzada
print("\n>>> Validação Cruzada (Regressão):")
cv_results = cross_validate(model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print(cv_results)

# 8. Preparar dados para gráficos
print(">>> Gerando gráficos...")
true_ratings = np.array([pred.r_ui for pred in predictions])
pred_ratings = np.array([pred.est for pred in predictions])
errors = np.abs(true_ratings - pred_ratings)

plt.figure(figsize=(8,6))
sns.scatterplot(x=true_ratings, y=pred_ratings, alpha=0.4)
plt.plot([0, 5], [0, 5], '--', color='red')
plt.title('Notas Reais vs. Notas Previstas (SVD)')
plt.xlabel('Nota Real')
plt.ylabel('Nota Prevista')
plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(errors, bins=30, kde=True, color='orange')
plt.title('Distribuição dos Erros Absolutos (|real - previsto|)')
plt.xlabel('Erro Absoluto')
plt.ylabel('Frequência')
plt.grid(True)
plt.show()

# 9. Métricas de Classificação
print(">>> Avaliando o modelo como classificação...")
bins = [0, 3, 6, 10]
labels = ['baixa', 'média', 'alta']
true_categories = pd.cut(true_ratings, bins=bins, labels=labels, include_lowest=True)
pred_categories = pd.cut(pred_ratings, bins=bins, labels=labels, include_lowest=True)

accuracy_cat = accuracy_score(true_categories, pred_categories)
precision_cat = precision_score(true_categories, pred_categories, average='weighted')
f1_cat = f1_score(true_categories, pred_categories, average='weighted')
recall_cat = recall_score(true_categories, pred_categories, average='weighted')

print("\n>>> Métricas de Classificação (Categorias):")
print("Accuracy:", round(accuracy_cat, 4))
print("Precision:", round(precision_cat, 4))
print("F1 Score:", round(f1_cat, 4))
print("Recall:", round(recall_cat, 4))

# Matriz de confusão
conf_matrix = confusion_matrix(true_categories, pred_categories, labels=labels)
conf_matrix_df = pd.DataFrame(conf_matrix, index=labels, columns=labels)
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues")
plt.title('Matriz de Confusão (Categorias de Rating)')
plt.xlabel('Predição')
plt.ylabel('Real')
plt.show()

# 10. Análise de resíduos
print(">>> Analisando resíduos...")
residuals = true_ratings - pred_ratings
plt.figure(figsize=(8,5))
plt.hist(residuals, bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Resíduos (Real - Previsto)')
plt.ylabel('Frequência')
plt.title('Histograma dos Resíduos')
plt.show()

plt.figure(figsize=(6,6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q plot dos Resíduos')
plt.show()

# 11. Recomendação de Filmes
def recomendar_filmes(usuario_id, model, baseRatings, baseMovies, top_n=10):
    print(f">>> Gerando recomendações para o usuário {usuario_id}...")
    filmes_avaliados = baseRatings[baseRatings['userId'] == usuario_id]['movieId'].unique()
    todos_filmes = baseRatings['movieId'].unique()
    filmes_nao_avaliados = [filme for filme in todos_filmes if filme not in filmes_avaliados]
    previsoes = [model.predict(usuario_id, filme_id) for filme_id in filmes_nao_avaliados]
    previsoes.sort(key=lambda x: x.est, reverse=True)
    top_filmes = previsoes[:top_n]
    recomendacoes = pd.DataFrame({
        'movieId': [pred.iid for pred in top_filmes],
        'Nota Prevista': [pred.est for pred in top_filmes]
    })
    recomendacoes = recomendacoes.merge(baseMovies, on='movieId')
    return recomendacoes[['movieId', 'title', 'Nota Prevista']]

# 12. Exemplo de uso da função de recomendação
usuario_exemplo = 1
recomendacoes_usuario = recomendar_filmes(usuario_exemplo, model, baseRatings, baseMovies, top_n=10)
print(f"\n>>> Top 10 filmes recomendados para o usuário {usuario_exemplo}:\n")
print(recomendacoes_usuario)