import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# =====================================================
# 1) ПОДГОТОВКА
# =====================================================
device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

# =====================================================
# 2) ЗАГРУЗКА ДАННЫХ
# =====================================================
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['MedHouseVal'] = data.target

# Удаляем цензурированные значения (5.00001)
df = df[df['MedHouseVal'] != 5.00001]

print(f"Размер данных: {df.shape}")

# =====================================================
# 3) УДАЛЕНИЕ ВЫБРОСОВ (IQR)
# =====================================================
Q1, Q3 = df['MedHouseVal'].quantile(0.25), df['MedHouseVal'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['MedHouseVal'] >= Q1 - 1.5 * IQR) & (df['MedHouseVal'] <= Q3 + 1.5 * IQR)]
print(f"После удаления выбросов: {df.shape}")

# =====================================================
# 4) ПРЕОБРАЗОВАНИЕ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ
# =====================================================
df['MedHouseVal'] = np.log(df['MedHouseVal'])

# =====================================================
# 5) УДАЛЕНИЕ КОРРЕЛИРУЮЩИХ ПРИЗНАКОВ
# =====================================================
df = df.drop(['AveBedrms', 'Longitude'], axis=1)

# =====================================================
# 6) ДОБАВЛЕНИЕ КВАДРАТИЧНЫХ ПРИЗНАКОВ
# =====================================================
df['HouseAge_sq'] = df['HouseAge'] ** 2
df['Population_sq'] = df['Population'] ** 2
df['MedInc_sq'] = df['MedInc'] ** 2

# =====================================================
# 7) ПОДГОТОВКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ
# =====================================================
X = df.drop('MedHouseVal', axis=1).values
y = df['MedHouseVal'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

# =====================================================
# 8) СОЗДАНИЕ И ОБУЧЕНИЕ МОДЕЛИ
# =====================================================
model = nn.Linear(X_train.shape[1], 1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)

epochs = 300
history = []

print("\nОбучение модели...")
for epoch in range(epochs):
    optimizer.zero_grad()
    loss = criterion(model(X_train_t), y_train_t)
    loss.backward()
    optimizer.step()
    history.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# =====================================================
# 9) ОЦЕНКА МОДЕЛИ И АНАЛИЗ ОСТАТКОВ
# =====================================================
model.eval()
with torch.no_grad():
    y_pred = model(X_test_t).cpu().numpy()
    y_true = y_test_t.cpu().numpy()

residuals = y_true - y_pred
r2 = r2_score(y_true, y_pred)

print(f"\nR² = {r2:.4f}")

# Финальная визуализация (4 графика)
plt.figure(figsize=(15, 10))

# 1. Процесс обучения
plt.subplot(2, 2, 1)
plt.plot(history)
plt.title("Процесс обучения")
plt.xlabel("Эпоха")
plt.ylabel("MSE Loss")
plt.grid(True)

# 2. Остатки vs Предсказания
plt.subplot(2, 2, 2)
plt.scatter(y_pred, residuals, alpha=0.3)
plt.axhline(0, color='red', linestyle='--')
plt.title("Анализ остатков")
plt.xlabel("Предсказанные значения")
plt.ylabel("Остатки")
plt.grid(True)

# 3. Гистограмма остатков
plt.subplot(2, 2, 3)
plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(0, color='red', linestyle='--')
plt.title("Распределение остатков")
plt.xlabel("Остатки")
plt.ylabel("Частота")

# 4. Корреляционная матрица
plt.subplot(2, 2, 4)
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', cbar=False)
plt.title("Корреляционная матрица")

plt.tight_layout()
plt.show()

# =====================================================
# 10) ИНТЕРПРЕТАЦИЯ
# =====================================================
print("\n" + "=" * 50)
print("ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ")
print("=" * 50)

print(f"\nКачество модели: R² = {r2:.4f} - " +
      ("хорошее" if r2 > 0.7 else "среднее" if r2 > 0.5 else "низкое"))

print(f"\nАнализ остатков:")
print(f"- Среднее остатков: {np.mean(residuals):.4f} " +
      ("(близко к 0 - хорошо)" if abs(np.mean(residuals)) < 0.01 else "(есть смещение)"))
print(f"- Стандартное отклонение: {np.std(residuals):.4f}")

print(f"\nВыводы:")
print(f"1. Модель {'хорошо' if r2 > 0.7 else 'удовлетворительно' if r2 > 0.5 else 'плохо'} справляется с задачей")
print(f"2. Квадратичные признаки {'помогли' if r2 > 0.6 else 'недостаточно эффективны'}")
print(
    f"3. {'Остатки распределены нормально' if abs(np.mean(residuals)) < 0.01 else 'Требуется дополнительная обработка данных'}")