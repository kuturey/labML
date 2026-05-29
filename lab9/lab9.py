import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from time import time
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings('ignore')


def read_and_preprocess(path):
    # Загрузка и первичная очистка данных
    data = pd.read_csv(path)
    print(f"Загружено: {data.shape[0]} строк, {data.shape[1]} столбцов")

    # Убираем технические поля, которые не влияют на класс объекта
    skip_cols = ['obj_ID', 'run_ID', 'rerun_ID', 'cam_col', 'field_ID',
                 'spec_obj_ID', 'plate', 'MJD', 'fiber_ID']
    skip_cols = [c for c in skip_cols if c in data.columns]
    data = data.drop(columns=skip_cols)
    print(f"Удалено {len(skip_cols)} технических колонок")

    # Убираем повторы
    before = len(data)
    data = data.drop_duplicates()
    print(f"Удалено дубликатов: {before - len(data)}")

    return data


def trim_outliers(df, cols, k=1.5):
    # Ограничиваем экстремальные значения через IQR
    df = df.copy()
    outlier_stats = []

    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        low = q1 - k * iqr
        high = q3 + k * iqr

        outlier_cnt = ((df[col] < low) | (df[col] > high)).sum()
        outlier_stats.append([col, outlier_cnt, f"{100*outlier_cnt/len(df):.1f}%"])

        df[col] = df[col].clip(low, high)

    print("\nОбнаружено выбросов:")
    print(pd.DataFrame(outlier_stats, columns=['Признак', 'Кол-во', 'Доля']).to_string(index=False))
    return df


def encode_target(df, target_name='class'):
    # Кодируем целевую переменную
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(df[target_name])
    print(f"\nЦелевые классы: {list(encoder.classes_)}")
    print("Распределение:")
    print(df[target_name].value_counts())
    return y_encoded, encoder


def reduce_dataset(X, y, max_size=3000):
    # Уменьшаем датасет для ускорения обучения SVM
    if len(X) <= max_size:
        return X, y

    X_small, _, y_small, _ = train_test_split(
        X, y, train_size=max_size, stratify=y, random_state=42
    )
    print(f"\nВзята подвыборка: {len(X_small)} объектов (из {len(X)})")
    return X_small, y_small


def build_svm_model():
    # Создает пайплайн с нормализацией и SVM
    return Pipeline([
        ('fill_na', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('svm', SVC(kernel='rbf', random_state=42, class_weight='balanced'))
    ])


def find_best_params(model, X_train, y_train, c_vals, gamma_vals):
    """Подбирает оптимальные C и gamma через кросс-валидацию"""
    param_grid = {'svm__C': c_vals, 'svm__gamma': gamma_vals}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    searcher = GridSearchCV(model, param_grid, cv=cv, scoring='accuracy',
                            n_jobs=-1, verbose=1)

    start = time()
    searcher.fit(X_train, y_train)
    print(f"Подбор параметров занял {time() - start:.1f} сек")

    return searcher


def draw_heatmap(grid_result, c_vals, gamma_vals):
    # Визуализирует результаты GridSearch в виде тепловой карты
    results = pd.DataFrame(grid_result.cv_results_)
    pivot = results.pivot_table(index='param_svm__C', columns='param_svm__gamma',
                                values='mean_test_score')

    plt.figure(figsize=(10, 7))
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlBu', center=0.95)
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.title('Зависимость качества от гиперпараметров SVM')
    plt.tight_layout()
    plt.show()


def main():
    print("\n" + "="*60)
    print("КЛАССИФИКАЦИЯ АСТРОНОМИЧЕСКИХ ОБЪЕКТОВ (SVM)")
    print("="*60 + "\n")

    # Загрузка
    data = read_and_preprocess("star_classification.csv")

    # Удаляем выбросы
    numeric_feats = data.select_dtypes(include=[np.number]).columns.tolist()
    if 'class' in numeric_feats:
        numeric_feats.remove('class')
    data = trim_outliers(data, numeric_feats, k=1.5)

    # Подготовка X и y
    X_full = data.drop('class', axis=1)
    y_full, label_enc = encode_target(data)

    # Уменьшаем датасет при необходимости
    X, y = reduce_dataset(X_full, y_full, max_size=3000)

    # Разделяем на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

    # Параметры для перебора
    c_range = [0.1, 0.5, 1, 5, 10, 50, 100]
    gamma_range = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

    # Обучение с подбором параметров
    svm = build_svm_model()
    search = find_best_params(svm, X_train, y_train, c_range, gamma_range)

    # Оценка на тесте
    y_pred = search.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ")
    print("="*60)
    print(f"Лучшие параметры: C={search.best_params_['svm__C']}, gamma={search.best_params_['svm__gamma']}")
    print(f"Лучшее CV качество: {search.best_score_:.4f}")
    print(f"Точность на тесте: {test_acc:.4f}")

    print("\n" + "-"*60)
    print("Детальная статистика по классам:")
    print("-"*60)
    print(classification_report(y_test, y_pred, target_names=label_enc.classes_))

    # Визуализация матрицы ошибок
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred,
                                            display_labels=label_enc.classes_,
                                            cmap='Greens', ax=ax)
    ax.set_title('Матрица ошибок')
    plt.tight_layout()
    plt.show()

    # Тепловая карта
    draw_heatmap(search, c_range, gamma_range)

    print("\n" + "="*60)
    print(f"Итог: достигнута точность {test_acc:.2%} на тестовой выборке")
    print("="*60)


if __name__ == "__main__":
    main()