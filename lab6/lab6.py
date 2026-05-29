import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 1. Реализация KNN
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        # Вычисляем евклидовы расстояния до всех обучающих объектов
        distances = [np.sqrt(np.sum((x - x_train) ** 2)) for x_train in self.X_train]
        # Находим k ближайших соседей
        k_indices = np.argsort(distances)[:self.k]
        k_labels = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]

# 2. Загрузка и подготовка данных Wine
wine = load_wine()
X = wine.data
y = wine.target

print("Датасет Wine")
print(f"Количество образцов: {X.shape[0]}")
print(f"Количество признаков: {X.shape[1]}")
print(f"Классы: {wine.target_names}")
print()

# Масштабирование признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Подбор оптимального k с помощью кросс-валидации
k_values = range(1, 16)
mean_scores = []

print("=" * 60)
print("Подбор оптимального k (5-кратная кросс-валидация)")
print("=" * 60)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for k in k_values:
    fold_scores = []

    for train_idx, val_idx in kf.split(X_scaled):
        X_train_fold = X_scaled[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X_scaled[val_idx]
        y_val_fold = y[val_idx]

        knn = KNN(k=k)
        knn.fit(X_train_fold, y_train_fold)
        y_pred_fold = knn.predict(X_val_fold)

        acc = accuracy_score(y_val_fold, y_pred_fold)
        fold_scores.append(acc)

    mean_acc = np.mean(fold_scores)
    mean_scores.append(mean_acc)

    print(f"k = {k:2d} | Средняя точность: {mean_acc:.4f} | Точность по фолдам: {[round(x, 4) for x in fold_scores]}")

best_k = k_values[np.argmax(mean_scores)]
print(f"\nЛучшее k: {best_k} (средняя точность = {max(mean_scores):.4f})")

# 4. График зависимости точности от k (метод локтя)
plt.figure(figsize=(10, 6))
plt.plot(k_values, mean_scores, marker='o', linestyle='--', color='blue', markersize=8)
plt.axvline(x=best_k, color='red', linestyle='--', label=f'Лучшее k = {best_k}')
plt.xlabel('k (количество соседей)', fontsize=12)
plt.ylabel('Средняя точность (accuracy)', fontsize=12)
plt.title('Выбор оптимального k для k-NN (Wine dataset)', fontsize=14)
plt.xticks(k_values)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 5. Финальное обучение и проверка на тестовой выборке
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

final_knn = KNN(k=best_k)
final_knn.fit(X_train, y_train)
y_pred = final_knn.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("Проверка на тестовой выборке")
print("=" * 60)
print(f"Лучшее k: {best_k}")
print(f"Точность на тесте: {test_accuracy:.4f}")

# 6. Визуализация границ классов (по первым двум признакам)
print("\n" + "=" * 60)
print("Визуализация границ классов (первые 2 признака)")
print("=" * 60)

# Берем только первые два признака
X_2d = X_scaled[:, :2]
knn_2d = KNN(k=best_k)
knn_2d.fit(X_2d, y)

# Создаем сетку точек для предсказания
x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                     np.arange(y_min, y_max, 0.05))

# Предсказываем класс для каждой точки сетки
Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Рисуем
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Set1)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap=plt.cm.Set1, edgecolor='k', s=50)
plt.xlabel(f'{wine.feature_names[0]} (стандартизированный)', fontsize=12)
plt.ylabel(f'{wine.feature_names[1]} (стандартизированный)', fontsize=12)
plt.title(f'Границы классов k-NN (k={best_k}) на Wine dataset', fontsize=14)
plt.grid(True, alpha=0.3)
plt.show()

# 7. Итоговые результаты
print("\n" + "=" * 60)
print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
print("=" * 60)
print(f"Датасет: Wine (3 класса вин)")
print(f"Оптимальное количество соседей k = {best_k}")
print(f"Средняя точность по кросс-валидации: {max(mean_scores):.4f}")
print(f"Точность на тестовой выборке: {test_accuracy:.4f}")
print(f"Визуализация границ выполнена по первым двум признакам")