import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, recall_score

# Загрузка данных
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Разделение данных с random_state=10
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=10, stratify=y
)

# Масштабирование признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Базовая модель для сравнения
model_default = LogisticRegression(C=1.0, solver='lbfgs', random_state=10)
model_default.fit(X_train_scaled, y_train)
y_pred_default = model_default.predict(X_test_scaled)

print("=== БАЗОВАЯ МОДЕЛЬ (C=1.0) ===")
print(confusion_matrix(y_test, y_pred_default))
print(classification_report(y_test, y_pred_default, target_names=data.target_names))
print(f"Recall для malignant: {recall_score(y_test, y_pred_default, pos_label=0):.4f}")

# 2. Модель с class_weight='balanced' - простая стратегия для минимизации FN
model_balanced = LogisticRegression(
    C=1.0,
    class_weight='balanced',  # Автоматически учитывает дисбаланс классов
    solver='lbfgs',
    random_state=10
)
model_balanced.fit(X_train_scaled, y_train)
y_pred_balanced = model_balanced.predict(X_test_scaled)

print("\n=== МОДЕЛЬ С CLASS_WEIGHT='BALANCED' ===")
print(confusion_matrix(y_test, y_pred_balanced))
print(classification_report(y_test, y_pred_balanced, target_names=data.target_names))
print(f"Recall для malignant: {recall_score(y_test, y_pred_balanced, pos_label=0):.4f}")

# 3. Модель с ручной настройкой весов (еще больше акцент на malignant)
model_weighted = LogisticRegression(
    C=1.0,
    class_weight={0: 3, 1: 1},  # Увеличиваем вес злокачественных в 3 раза
    solver='lbfgs',
    random_state=10
)
model_weighted.fit(X_train_scaled, y_train)
y_pred_weighted = model_weighted.predict(X_test_scaled)

print("\n=== МОДЕЛЬ С CLASS_WEIGHT (malignant: 3, benign: 1) ===")
print(confusion_matrix(y_test, y_pred_weighted))
print(classification_report(y_test, y_pred_weighted, target_names=data.target_names))
print(f"Recall для malignant: {recall_score(y_test, y_pred_weighted, pos_label=0):.4f}")

# 4. Модель с L1 регуляризацией (отбор признаков)
model_l1 = LogisticRegression(
    C=0.5,
    penalty='l1',
    solver='saga',
    class_weight='balanced',
    max_iter=5000,
    random_state=10
)
model_l1.fit(X_train_scaled, y_train)
y_pred_l1 = model_l1.predict(X_test_scaled)

print("\n=== МОДЕЛЬ С L1 РЕГУЛЯРИЗАЦИЕЙ ===")
print(confusion_matrix(y_test, y_pred_l1))
print(classification_report(y_test, y_pred_l1, target_names=data.target_names))
print(f"Recall для malignant: {recall_score(y_test, y_pred_l1, pos_label=0):.4f}")
print(f"Количество ненулевых признаков: {np.sum(model_l1.coef_[0] != 0)} из {X.shape[1]}")

# 5. Анализ важности признаков для лучшей модели
# Выбираем модель с наибольшим recall для malignant
models = {
    'Базовая': recall_score(y_test, y_pred_default, pos_label=0),
    'Balanced': recall_score(y_test, y_pred_balanced, pos_label=0),
    'Weighted (3:1)': recall_score(y_test, y_pred_weighted, pos_label=0),
    'L1': recall_score(y_test, y_pred_l1, pos_label=0)
}

print("\n=== СРАВНЕНИЕ RECALL ДЛЯ MALIGNANT ===")
for name, recall in models.items():
    print(f"{name}: {recall:.4f}")

# Выбираем лучшую модель по recall
best_model_name = max(models, key=models.get)
print(f"\nЛучшая модель по recall: {best_model_name}")

# Для лучшей модели показываем важные признаки
if best_model_name == 'L1':
    best_model = model_l1
else:
    best_model = model_balanced  # или другая модель

print("\n=== ВАЖНЫЕ ПРИЗНАКИ ДЛЯ ОПРЕДЕЛЕНИЯ ЗЛОКАЧЕСТВЕННЫХ ОПУХОЛЕЙ ===")
weights = pd.Series(best_model.coef_[0], index=data.feature_names)
# Положительные веса = признак указывает на злокачественность
top_features = weights.sort_values(ascending=False).head(10)
print(top_features)