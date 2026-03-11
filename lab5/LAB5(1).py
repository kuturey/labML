import warnings
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier

warnings.filterwarnings("ignore")

# Загрузка данных
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Масштабирование признаков
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

print("=" * 60)
print("ЗАДАНИЕ 1: Влияние параметра C на зануление весов")
print("=" * 60)

Cs = [0.001, 0.01, 0.1, 1, 10]

for C in Cs:
    # Модель с L1 регуляризацией
    base_model = LogisticRegression(
        penalty='l1',
        C=C,
        solver='liblinear',
        max_iter=1000,
        random_state=42
    )
    model = OneVsRestClassifier(base_model)
    model.fit(X_train_std, y_train)

    # Предсказание и точность
    y_pred = model.predict(X_test_std)
    accuracy = accuracy_score(y_test, y_pred)

    # Собираем все веса в одну матрицу
    coef_matrix = np.vstack([est.coef_.ravel() for est in model.estimators_])

    # Считаем нулевые веса
    zero_weights = np.sum(coef_matrix == 0)
    total_weights = coef_matrix.size

    print(f"\nC = {C}:")
    print(f"  Нулевых весов: {zero_weights} из {total_weights} ({zero_weights / total_weights * 100:.1f}%)")
    print(f"  Точность: {accuracy:.4f}")
    print(f"  Веса:\n{coef_matrix}")
    print("-" * 60)

print("\n" + "=" * 60)
print("ЗАДАНИЕ 2: Сравнение L1 и L2 при C=0.01")
print("=" * 60)

C_small = 0.01

# L1 модель
base_l1 = LogisticRegression(
    penalty='l1',
    C=C_small,
    solver='liblinear',
    max_iter=1000,
    random_state=42
)
model_l1 = OneVsRestClassifier(base_l1)
model_l1.fit(X_train_std, y_train)

# L2 модель
model_l2 = LogisticRegression(
    penalty='l2',
    C=C_small,
    solver='lbfgs',
    max_iter=1000,
    random_state=42
)
model_l2.fit(X_train_std, y_train)

# Предсказания
y_pred_l1 = model_l1.predict(X_test_std)
y_pred_l2 = model_l2.predict(X_test_std)

# Собираем веса L1
l1_weights = np.vstack([est.coef_.ravel() for est in model_l1.estimators_])
l2_weights = model_l2.coef_

print(f"\nL1 веса (C={C_small}):")
print(l1_weights)
print(f"\nКоличество нулевых весов в L1: {np.sum(l1_weights == 0)}")
print(f"Точность L1: {accuracy_score(y_test, y_pred_l1):.4f}")

print(f"\nL2 веса (C={C_small}):")
print(l2_weights)
print(f"\nКоличество нулевых весов в L2: {np.sum(l2_weights == 0)}")
print(f"Точность L2: {accuracy_score(y_test, y_pred_l2):.4f}")

print("\n" + "=" * 60)
print("ВЫВОДЫ:")
print("=" * 60)
print("1. L1 регуляризация зануляет веса (создает разреженные решения)")
print("2. L2 регуляризация равномерно уменьшает все веса")
print("3. При малом C (сильная регуляризация) L1 зануляет больше признаков")
print("4. L2 веса распределены более равномерно между признаками")
print("5. L1 создает резкий контраст между важными и неважными признаками")
print("6. При C=0.01 точность L2 может быть выше, т.к. L1 теряет информацию")