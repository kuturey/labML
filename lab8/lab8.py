import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Определяем, на чём будем считать (чтобы ускорить, если есть GPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Вычислительное устройство: {device}")


# Загружаем данные и смотрим на них
df = pd.read_csv("weatherAUS.csv")
print(f"Размер датасета: {df.shape}")
print("\nПервые записи:")
print(df.head())
print("\nИнформация о данных:")
print(df.info())


# Оставляем только те строки, где известно, был дождь или нет
df = df.dropna(subset=["RainTomorrow"]).copy()
# Превращаем ответы в числа
df["RainTomorrow"] = df["RainTomorrow"].map({"No": 0, "Yes": 1})

# Проверяем, не слишком ли сильно дисбаланс классов
print("\nСколько примеров каждого класса:")
print(df["RainTomorrow"].value_counts())
print(f"Доля дождливых дней: {df['RainTomorrow'].mean():.2%}")


# Из даты вытаскиваем полезные компоненты
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day
# Исходная дата больше не нужна
df.drop("Date", axis=1, inplace=True)


# Отделяем признаки от целевой переменной
X = df.drop("RainTomorrow", axis=1)
y = df["RainTomorrow"]
print(f"\nМатрица признаков X: {X.shape}")
print(f"Вектор ответов y: {y.shape}")


# Делим данные на обучающую и тестовую выборки (стратификация сохраняет пропорцию классов)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=123, stratify=y
)
print(f"\nОбучающая выборка: {X_train.shape}")
print(f"Тестовая выборка: {X_test.shape}")


# Определяем, какие столбцы числовые, а какие категориальные
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object", "string"]).columns

print("\nЧисловые признаки:")
print(list(numeric_features))
print("Категориальные признаки:")
print(list(categorical_features))


# Для числовых признаков: пропуски заполняем медианой
numeric_transformer = SimpleImputer(strategy="median")

# Для категориальных: пропуски заполняем самой частой категорией, потом one-hot кодирование
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False))
])

# Объединяем обработку в один трансформер
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# Модель 1: дерево с энтропией (похоже на C4.5)
model_entropy = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(criterion="entropy", random_state=123))
])

# Модель 2: дерево с индексом Джини (классический CART)
model_gini = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(criterion="gini", random_state=123))
])

# Модель 3: случайный лес из 100 деревьев
model_rf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=123, n_jobs=-1))
])


# Обучаем все три модели
print("\n" + "-"*50)
print("Начинаем обучение")
print("-"*50)

model_entropy.fit(X_train, y_train)
print("Дерево с энтропией — готово")

model_gini.fit(X_train, y_train)
print("Дерево с Джини — готово")

model_rf.fit(X_train, y_train)
print("Случайный лес — готово")


# Делаем предсказания на тестовой выборке
pred_entropy = model_entropy.predict(X_test)
pred_gini = model_gini.predict(X_test)
pred_rf = model_rf.predict(X_test)


# Считаем точность каждой модели
acc_entropy = accuracy_score(y_test, pred_entropy)
acc_gini = accuracy_score(y_test, pred_gini)
acc_rf = accuracy_score(y_test, pred_rf)

print("\n" + "-"*50)
print("Точность моделей")
print("-"*50)
print(f"Decision Tree (entropy):  {acc_entropy:.4f}")
print(f"Decision Tree (gini):     {acc_gini:.4f}")
print(f"Random Forest:            {acc_rf:.4f}")


# Выводим подробные отчёты для каждой модели
print("\n" + "-"*50)
print("Отчёт для дерева с энтропией")
print("-"*50)
print("Матрица ошибок:")
print(confusion_matrix(y_test, pred_entropy))
print("Classification report:")
print(classification_report(y_test, pred_entropy))

print("\n" + "-"*50)
print("Отчёт для дерева с Джини")
print("-"*50)
print("Матрица ошибок:")
print(confusion_matrix(y_test, pred_gini))
print("Classification report:")
print(classification_report(y_test, pred_gini))

print("\n" + "-"*50)
print("Отчёт для случайного леса")
print("-"*50)
print("Матрица ошибок:")
print(confusion_matrix(y_test, pred_rf))
print("Classification report:")
print(classification_report(y_test, pred_rf))


# Собираем результаты в таблицу и сортируем
results = pd.DataFrame({
    "Модель": ["C4.5 (entropy)", "CART (gini)", "Random Forest"],
    "Accuracy": [acc_entropy, acc_gini, acc_rf]
})
results = results.sort_values(by="Accuracy", ascending=False)

print("\n" + "-"*50)
print("Итоговое сравнение")
print("-"*50)
print(results.to_string(index=False))


# Рисуем график для наглядности
plt.figure(figsize=(8, 5))
bars = plt.bar(results["Модель"], results["Accuracy"], color=['#2ecc71', '#3498db', '#e74c3c'])
plt.title("Сравнение точности моделей", fontsize=14)
plt.xlabel("Модель", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.ylim(0, 1)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{height:.4f}', ha='center', va='bottom', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# Финальный вывод
best_model = results.iloc[0]["Модель"]
best_acc = results.iloc[0]["Accuracy"]

print("\n" + "-"*50)
print("Вывод")
print("-"*50)
print(f"Лучше всего показала себя модель: {best_model}")
print(f"Её точность: {best_acc:.4f}")