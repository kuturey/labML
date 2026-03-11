import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def calculate_root_mse(actual, predicted):
    """Корень из среднеквадратичной ошибки"""
    mse_value = mean_squared_error(actual, predicted)
    return np.sqrt(mse_value)


def print_model_results(experiment_name, y_true, y_pred, additional_info=""):
    """Вывод метрик качества модели"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = calculate_root_mse(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n▶ {experiment_name}")
    if additional_info:
        print(f"  {additional_info}")
    print(f"  MSE : {mse:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  MAE : {mae:.3f}")
    print(f"  R²  : {r2:.3f}")

    return {
        'name': experiment_name,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


# Загрузка данных
file_path = "house_price_regression_dataset.csv"
data_frame = pd.read_csv(file_path)
print("Размер датасета:", data_frame.shape)

target_variable = "House_Price"
features = data_frame.drop(columns=[target_variable])
target = data_frame[target_variable]

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=10
)
print("Обучающая выборка:", X_train.shape)
print("Тестовая выборка :", X_test.shape)

# Анализ корреляций
correlations = data_frame.corr(numeric_only=True)[target_variable].sort_values(
    key=lambda x: abs(x)
)
print("\nКорреляция с целевой переменной:")
print(correlations)

# Список для хранения результатов
experiment_results = []

# Эксперимент A: Ридж-регрессия с масштабированием
scaler_a = StandardScaler()
X_train_a = scaler_a.fit_transform(X_train)
X_test_a = scaler_a.transform(X_test)

ridge_model = Ridge(alpha=4)
ridge_model.fit(X_train_a, y_train)
pred_a = ridge_model.predict(X_test_a)

experiment_results.append(
    print_model_results("Ridge регрессия (α=4) + StandardScaler", y_test, pred_a)
)

# Эксперимент B: Только два самых важных признака
important_features = ["Square_Footage", "Lot_Size"]
reduced_df = data_frame[important_features + [target_variable]].copy()

X_reduced = reduced_df[important_features]
y_reduced = reduced_df[target_variable]

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_reduced, y_reduced, test_size=0.2, random_state=10
)

lr_reduced = LinearRegression()
lr_reduced.fit(X_train_b, y_train_b)
pred_b = lr_reduced.predict(X_test_b)

experiment_results.append(
    print_model_results(
        "Линейная регрессия (2 признака)",
        y_test_b,
        pred_b,
        f"Использованы: {important_features}"
    )
)

# Эксперимент C: Стандартная линейная регрессия без масштабирования
lr_simple = LinearRegression()
lr_simple.fit(X_train, y_train)
pred_c = lr_simple.predict(X_test)

experiment_results.append(
    print_model_results("LinearRegression (без масштабирования)", y_test, pred_c)
)

# Эксперимент D: Удаление слабых признаков
weak_features = ["Num_Bathrooms", "Neighborhood_Quality", "Num_Bedrooms"]
clean_df = data_frame.drop(columns=weak_features)

X_clean = clean_df.drop(columns=[target_variable])
y_clean = clean_df[target_variable]

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_clean, y_clean, test_size=0.2, random_state=10
)

lr_clean = LinearRegression()
lr_clean.fit(X_train_d, y_train_d)
pred_d = lr_clean.predict(X_test_d)

experiment_results.append(
    print_model_results(
        "LinearRegression (удалены слабые признаки)",
        y_test_d,
        pred_d,
        f"Удалено: {weak_features}"
    )
)

# Эксперимент E: Линейная регрессия со стандартизацией
scaler_e = StandardScaler()
X_train_e = scaler_e.fit_transform(X_train)
X_test_e = scaler_e.transform(X_test)

lr_scaled = LinearRegression()
lr_scaled.fit(X_train_e, y_train)
pred_e = lr_scaled.predict(X_test_e)

experiment_results.append(
    print_model_results("LinearRegression + StandardScaler", y_test, pred_e)
)

# Формирование итоговой таблицы
summary_df = pd.DataFrame(experiment_results)

# Форматирование результатов
summary_df['mse'] = summary_df['mse'].round(0).astype(int)
summary_df['rmse'] = summary_df['rmse'].round(0).astype(int)
summary_df['mae'] = summary_df['mae'].round(0).astype(int)
summary_df['r2'] = summary_df['r2'].round(4)

# Переименование колонок
summary_df.columns = ['Эксперимент', 'MSE', 'RMSE', 'MAE', 'R²']

# Сортировка по качеству (лучший RMSE - сверху)
summary_df = summary_df.sort_values('RMSE').reset_index(drop=True)

print("\n" + "=" * 60)
print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ (по возрастанию RMSE)")
print("=" * 60)
print(summary_df.to_string(index=False))