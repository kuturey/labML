import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

# Фиксируем случайность для воспроизводимости
np.random.seed(123)

# ЧАСТЬ 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ

print("\n" + "=" * 60)
print("ЗАГРУЗКА ДАННЫХ")
print("=" * 60)

# Загружаем встроенный датасет
cancer = load_breast_cancer()
features = cancer.data  # матрица признаков (569x30)
labels = cancer.target  # целевая переменная (0=злокачеств., 1=доброкач.)

print(f"Размер выборки: {features.shape[0]} объектов")
print(f"Количество признаков: {features.shape[1]}")
print(f"Классы: 0 (злокачественная) - {sum(labels == 0)} шт., 1 (доброкачественная) - {sum(labels == 1)} шт.")

# Нормализация данных (метрические методы чувствительны к масштабу)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# ВЫБОР ДВУХ НАИБОЛЕЕ ЗНАЧИМЫХ ПРИЗНАКОВ
selector = SelectKBest(score_func=f_classif, k=2)
features_selected = selector.fit_transform(features_scaled, labels)

selected_indices = selector.get_support(indices=True)
selected_features = cancer.feature_names[selected_indices]

print(f"\nВыбранные признаки:")
print(f"  1. {selected_features[0]}")
print(f"  2. {selected_features[1]}")

# Разделение на обучающую (70%) и тестовую (30%) выборки
X_train, X_test, y_train, y_test = train_test_split(
    features_selected, labels, test_size=0.3, stratify=labels, random_state=42
)

print(f"\nОбучающая выборка: {X_train.shape[0]} объектов")
print(f"Тестовая выборка: {X_test.shape[0]} объектов")

# Визуализация обучающей выборки
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                      cmap="coolwarm", edgecolor="k", alpha=0.7)
plt.legend(handles=scatter.legend_elements()[0],
           labels=["Malignant (0)", "Benign (1)"])
plt.title("Breast Cancer Dataset (2 selected features)")
plt.xlabel(selected_features[0])
plt.ylabel(selected_features[1])
plt.show()


# ЧАСТЬ 2. МЕТОД ПАРЗЕНОВСКОГО ОКНА

class ParzenWindowClassifier:
    # Классификатор на основе парзеновского окна с гауссовым ядром

    def __init__(self, h=0.5):
        self.h = h  # ширина окна (параметр h)
        self.X_train = None
        self.y_train = None
        self.classes = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.classes = np.unique(y)

    def _gaussian_kernel(self, distance):
        # Гауссово ядро K(d) = 1/√(2π) * exp(-d²/2)
        return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * distance ** 2)

    def predict_proba(self, X):
        # Оцениваем плотность вероятности для каждого класса
        probabilities = np.zeros((X.shape[0], len(self.classes)))

        for i, sample in enumerate(X):
            for j, cls in enumerate(self.classes):
                # Берём точки обучающей выборки текущего класса
                class_points = self.X_train[self.y_train == cls]
                if len(class_points) == 0:
                    continue
                # Считаем расстояния
                distances = np.linalg.norm(class_points - sample, axis=1)
                # Суммируем вклады ядер
                probabilities[i, j] = np.sum(self._gaussian_kernel(distances / self.h))

        return probabilities

    def predict(self, X):
        # Предсказываем класс с максимальной вероятностью
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]


# ЧАСТЬ 3. МЕТОД ПОТЕНЦИАЛЬНЫХ ФУНКЦИЙ

class PotentialFunctionClassifier:
    # Классификатор на основе метода потенциальных функций

    def __init__(self, h=1.0, epochs=15):
        self.h = h  # ширина потенциала (параметр h)
        self.epochs = epochs  # максимальное число эпох
        self.support_vectors_ = []  # опорные векторы
        self.charges_ = []  # веса (заряды) ±1

    def _kernel(self, u):
        # Гауссово ядро: K(u) = exp(-u²)
        return np.exp(-u ** 2)

    def _decision_function_single(self, x):
        if not self.support_vectors_:
            return 0.0

        support_vectors = np.array(self.support_vectors_)
        charges = np.array(self.charges_)
        distances = np.linalg.norm(support_vectors - x, axis=1)
        return np.sum(charges * self._kernel(distances / self.h))

    def fit(self, X, y):
        # Переводим метки в формат ±1 (0  -1, 1  1)
        y_signed = np.where(y == 0, -1, 1)
        n_samples = X.shape[0]

        for epoch in range(self.epochs):
            errors = 0
            for i in range(n_samples):
                x_curr = X[i]
                y_true = y_signed[i]

                decision_val = self._decision_function_single(x_curr)
                y_pred = 1 if decision_val > 0 else -1

                if y_pred != y_true:
                    self.support_vectors_.append(x_curr.copy())
                    self.charges_.append(y_true)
                    errors += 1

            print(f"  Эпоха {epoch + 1}: добавлено {errors} потенциалов, всего = {len(self.charges_)}")

            if errors == 0:
                print(f"  >> Сходимость достигнута на эпохе {epoch + 1}")
                break

    def predict(self, X):
        predictions = []
        for x in X:
            score = self._decision_function_single(x)
            predictions.append(1 if score > 0 else 0)
        return np.array(predictions)


# ЧАСТЬ 4. ПОДБОР ГИПЕРПАРАМЕТРА H

print("\n" + "=" * 60)
print("ПОДБОР ПАРАМЕТРА h")
print("=" * 60)

# Разбиваем обучение на подвыборки (80% обучение, 20% валидация)
X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=123
)

# Значения ширины окна для перебора
h_values = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

# Подбор для парзеновского окна
print("\n--- Парзеновское окно ---")
best_h_parzen = None
best_acc_parzen = 0.0

for h in h_values:
    model = ParzenWindowClassifier(h=h)
    model.fit(X_train_sub, y_train_sub)
    acc = accuracy_score(y_val, model.predict(X_val))
    print(f"  h = {h:.2f} → точность на валидации = {acc:.4f}")
    if acc > best_acc_parzen:
        best_acc_parzen = acc
        best_h_parzen = h

print(f"\n  >> Лучший h = {best_h_parzen} (точность = {best_acc_parzen:.4f})")

# Подбор для метода потенциальных функций
print("\n--- Метод потенциальных функций ---")
best_h_potential = None
best_acc_potential = 0.0

for h in h_values:
    model = PotentialFunctionClassifier(h=h, epochs=20)
    model.fit(X_train_sub, y_train_sub)
    acc = accuracy_score(y_val, model.predict(X_val))
    print(f"  h = {h:.2f} → точность = {acc:.4f}, опорных векторов = {len(model.charges_)}")
    if acc > best_acc_potential:
        best_acc_potential = acc
        best_h_potential = h

print(f"\n  >> Лучший h = {best_h_potential} (точность = {best_acc_potential:.4f})")

# ЧАСТЬ 5. ФИНАЛЬНОЕ ОБУЧЕНИЕ И ТЕСТИРОВАНИЕ

print("\n" + "=" * 60)
print("ФИНАЛЬНОЕ ТЕСТИРОВАНИЕ")
print("=" * 60)

# Обучаем на всей обучающей выборке с лучшими параметрами
parzen_best = ParzenWindowClassifier(h=best_h_parzen)
parzen_best.fit(X_train, y_train)

potential_best = PotentialFunctionClassifier(h=best_h_potential, epochs=20)
potential_best.fit(X_train, y_train)

# Предсказания на тестовой выборке
y_pred_parzen = parzen_best.predict(X_test)
y_pred_potential = potential_best.predict(X_test)

# Считаем точность
acc_parzen = accuracy_score(y_test, y_pred_parzen)
acc_potential = accuracy_score(y_test, y_pred_potential)

print(f"Parzen Window Classifier:      точность = {acc_parzen:.4f}")
print(f"Potential Function Classifier: точность = {acc_potential:.4f}")

# Сравнение результатов
print("\n--- Сравнение ---")
if acc_parzen > acc_potential:
    print(f"Лучше сработало Парзеновское окно (+{acc_parzen - acc_potential:.4f})")
elif acc_potential > acc_parzen:
    print(f"Лучше сработал метод потенциальных функций (+{acc_potential - acc_parzen:.4f})")
else:
    print("Методы показали одинаковый результат")


# ЧАСТЬ 6. ВИЗУАЛИЗАЦИЯ ГРАНИЦ РЕШЕНИЙ

def plot_decision_boundary(clf, X, y, title=""):
    # Определяем границы сетки
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # Создаём сетку
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.05),
        np.arange(y_min, y_max, 0.05)
    )

    # Предсказываем класс для каждой точки сетки
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(grid)
    Z = Z.reshape(xx.shape)

    # Рисуем
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(["#FFAAAA", "#AAAAFF"]))
    plt.scatter(X[:, 0], X[:, 1], c=y,
                cmap=ListedColormap(["#FF0000", "#0000FF"]),
                edgecolors="k", alpha=0.7)
    plt.title(title, fontsize=12)
    plt.xlabel(selected_features[0], fontsize=10)
    plt.ylabel(selected_features[1], fontsize=10)


# Создаём график
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plot_decision_boundary(parzen_best, X_test, y_test,
                       f"Parzen Window (h = {best_h_parzen})")

plt.subplot(1, 2, 2)
plot_decision_boundary(potential_best, X_test, y_test,
                       f"Potential Functions (h = {best_h_potential})")

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("ВЫПОЛНЕНО")
print("=" * 60)

# Дополнительный вывод информации о лучших моделях
print("\n" + "=" * 60)
print("ИТОГОВЫЙ ВЫВОД")
print("=" * 60)
print(f"Лучшее h для Parzen Window: {best_h_parzen}")
print(f"Лучшее h для Potential Function: {best_h_potential}")
print(f"Точность Parzen Window на тесте: {acc_parzen:.4f}")
print(f"Точность Potential Function на тесте: {acc_potential:.4f}")
print(f"Выбранные признаки: {selected_features[0]} и {selected_features[1]}")