import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

# Определение устройства для возможного использования в будущем
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Вычисление будет использовать: {device}")

# Загрузка и первичный анализ датасета Wine
wine = load_wine()
X = wine.data
y = wine.target

wine_data = pd.DataFrame(X, columns=wine.feature_names)
wine_data["target"] = y

print(f"Размер исходных данных: {wine_data.shape}")
print("\nПервые 5 строк:")
print(wine_data.head())

print("\nИнформация о датасете:")
print(wine_data.info())

print(f"\nНазвания классов: {list(wine.target_names)}")

print("\nРаспределение по классам:")
print(wine_data["target"].value_counts().sort_index())

# Отделение признаков от целевой переменной
X = wine_data.drop("target", axis=1)
y = wine_data["target"]

print(f"\nМатрица признаков: {X.shape}")
print(f"Вектор меток: {y.shape}")

# Нормализация данных (обязательный шаг перед кластеризацией)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Размер после масштабирования: {X_scaled.shape}")

# Сжатие до 2 измерений для визуализации
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"\nРазмер после PCA: {X_pca.shape}")
print("Доля объяснённой дисперсии по компонентам:")
print(pca.explained_variance_ratio_)
print(f"Суммарная объяснённая дисперсия: {pca.explained_variance_ratio_.sum():.4f}")

# Визуализация исходных классов в PCA-пространстве
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis")
plt.title("Визуализация Wine dataset после PCA")
plt.xlabel("Первая главная компонента")
plt.ylabel("Вторая главная компонента")
plt.colorbar(scatter, label="Исходный класс вина")
plt.show()

# Подбор оптимального k для KMeans
k_candidates = [2, 3, 4, 5]
kmeans_performance = []

for k in k_candidates:
    clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = clusterer.fit_predict(X_scaled)

    silhouette = silhouette_score(X_scaled, labels)
    ari = adjusted_rand_score(y, labels)
    nmi = normalized_mutual_info_score(y, labels)

    kmeans_performance.append({
        "k": k,
        "silhouette": silhouette,
        "ari": ari,
        "nmi": nmi,
        "inertia": clusterer.inertia_
    })

results_kmeans = pd.DataFrame(kmeans_performance).sort_values("ari", ascending=False)

print("\nРезультаты подбора параметров для KMeans")
print("-" * 60)
print(results_kmeans.to_string(index=False))

# Обучение лучшей модели KMeans
best_k = int(results_kmeans.iloc[0]["k"])
best_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans_labels = best_kmeans.fit_predict(X_scaled)
kmeans_centers_pca = pca.transform(best_kmeans.cluster_centers_)

print("\nЛучшая конфигурация KMeans")
print("-" * 60)
print(f"Оптимальное число кластеров: {best_k}")
print(f"Silhouette Score: {results_kmeans.iloc[0]['silhouette']:.4f}")
print(f"Adjusted Rand Index: {results_kmeans.iloc[0]['ari']:.4f}")
print(f"Normalized Mutual Info: {results_kmeans.iloc[0]['nmi']:.4f}")

# Визуализация результатов KMeans
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap="viridis")
plt.scatter(kmeans_centers_pca[:, 0], kmeans_centers_pca[:, 1],
            marker="X", s=200, c="red", label="Центроиды")
plt.title(f"KMeans кластеризация (k={best_k})")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend()
plt.show()

# Поиск оптимальных параметров для DBSCAN
eps_candidates = [1.5, 1.7, 2.0, 2.2, 2.5, 3.0]
min_samples_candidates = [3, 5, 7, 10]
dbscan_performance = []

for eps in eps_candidates:
    for min_samples in min_samples_candidates:
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        labels = clusterer.fit_predict(X_scaled)

        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        noise_count = (labels == -1).sum()

        silhouette = silhouette_score(X_scaled, labels) if n_clusters > 1 else -1
        ari = adjusted_rand_score(y, labels)
        nmi = normalized_mutual_info_score(y, labels)

        dbscan_performance.append({
            "eps": eps,
            "min_samples": min_samples,
            "clusters": n_clusters,
            "noise": noise_count,
            "silhouette": silhouette,
            "ari": ari,
            "nmi": nmi
        })

results_dbscan = pd.DataFrame(dbscan_performance).sort_values("ari", ascending=False)

print("\nРезультаты подбора параметров для DBSCAN")
print("-" * 60)
print(results_dbscan.head(10).to_string(index=False))

# Обучение лучшей модели DBSCAN
best_eps = results_dbscan.iloc[0]["eps"]
best_min_samples = int(results_dbscan.iloc[0]["min_samples"])
best_dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
dbscan_labels = best_dbscan.fit_predict(X_scaled)

print("\nЛучшая конфигурация DBSCAN")
print("-" * 60)
print(f"Оптимальный eps: {best_eps}")
print(f"Оптимальный min_samples: {best_min_samples}")
print(f"Найдено кластеров: {int(results_dbscan.iloc[0]['clusters'])}")
print(f"Точек-выбросов (шум): {int(results_dbscan.iloc[0]['noise'])}")
print(f"Silhouette Score: {results_dbscan.iloc[0]['silhouette']:.4f}")
print(f"Adjusted Rand Index: {results_dbscan.iloc[0]['ari']:.4f}")
print(f"Normalized Mutual Info: {results_dbscan.iloc[0]['nmi']:.4f}")

# Визуализация результатов DBSCAN
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap="viridis")
plt.title(f"DBSCAN кластеризация (eps={best_eps}, min_samples={best_min_samples})")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(scatter, label="ID кластера")
plt.show()

# Сравнительный анализ двух алгоритмов
comparison = pd.DataFrame({
    "Algorithm": ["KMeans", "DBSCAN"],
    "Parameters": [f"k={best_k}", f"eps={best_eps}, min_samples={best_min_samples}"],
    "Silhouette": [results_kmeans.iloc[0]["silhouette"], results_dbscan.iloc[0]["silhouette"]],
    "ARI": [results_kmeans.iloc[0]["ari"], results_dbscan.iloc[0]["ari"]],
    "NMI": [results_kmeans.iloc[0]["nmi"], results_dbscan.iloc[0]["nmi"]]
}).sort_values("ARI", ascending=False)

print("\nСравнение алгоритмов кластеризации")
print("-" * 60)
print(comparison.to_string(index=False))

# Визуализация сравнения
plt.figure(figsize=(8, 5))
bars = plt.bar(comparison["Algorithm"], comparison["ARI"], color=['steelblue', 'coral'])
plt.title("Сравнение качества кластеризации (метрика ARI)")
plt.xlabel("Алгоритм")
plt.ylabel("Adjusted Rand Index")
plt.ylim(0, 1)
for bar, ari in zip(bars, comparison["ARI"]):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             f"{ari:.3f}", ha='center', fontweight='bold')
plt.show()

# Финальный вердикт
best_model = comparison.iloc[0]["Algorithm"]
best_score = comparison.iloc[0]["ARI"]

print("\nИтоговое заключение")
print("-" * 60)
print(f"Лучший алгоритм: {best_model}")
print(f"Достигнутое значение ARI: {best_score:.4f}")