import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# 1. Загрузка данных "House Prices" (полупустой датасет)
df = pd.read_csv('house_prices.csv')
print(f'Размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов')

# Убираем столбец Id, он не несет смысла
df = df.drop('Id', axis='columns')

# 2. Выводим Топ-15 столбцов с наибольшим количеством пропусков
print('\n=== Топ-15 самых "дырявых" столбцов ДО заполнения ===')
missing = df.isnull().sum()
print(missing[missing > 0].sort_values(ascending=False).head(15))

# 3. Заполнение пропущенных значений
# Проходимся по всем столбцам и заполняем пропуски
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype == 'object':
            # Категориальные заполняем модой (самым частым значением)
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            # Числовые заполняем медианой
            df[col].fillna(df[col].median(), inplace=True)

print('\n=== Пропущенные значения ПОСЛЕ заполнения ===')
print(f'Осталось пустых ячеек во всем датасете: {df.isnull().sum().sum()}')

# 4. Нормализация числовых данных (MinMaxScaler)
numeric_cols = df.select_dtypes(include='number').columns.drop('SalePrice')
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
print('\n=== Данные после нормализации (первые 5 строк) ===')
display(df[numeric_cols].head())

# 5. Преобразование категориальных данных (One-Hot Encoding)
object_cols = df.select_dtypes(include='object').columns.tolist()
print(f'\nКатегориальных столбцов для OHE: {len(object_cols)}')
df = pd.get_dummies(df, columns=object_cols, drop_first=True)
print(f'\nИтоговый размер датасета после OHE: {df.shape[0]} строк, {df.shape[1]} столбцов')

# Разбиение на обучающую и тестовую выборки
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
print(f'\nОбучающая выборка: {train_df.shape}')
print(f'Тестовая выборка: {test_df.shape}')

# Сохранение результата
df.to_csv('processed_house_prices.csv', index=False)
print('\nОбработанный датасет сохранён в processed_house_prices.csv')
