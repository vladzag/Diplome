import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Функция для анализа качества данных
def analyze_data_quality(dataframe):
    report = {}

    # Пропущенные значения
    report['Пропущенные значения'] = dataframe.isnull().sum()

    # Уникальные значения
    report['Уникальные значения'] = dataframe.nunique()

    # Дубликаты
    report['Дубликаты'] = dataframe.duplicated().sum()

    # Аномалии (выбросы) с использованием межквартильного размаха
    anomalies = {}
    for column in dataframe.select_dtypes(include=[np.number]):
        Q1 = dataframe[column].quantile(0.25)
        Q3 = dataframe[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        anomalies[column] = ((dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)).sum()
    report['Аномалии'] = anomalies

    return report


# Очистка данных
def clean_data(dataframe):
    df_cleaned = dataframe.copy()

    # Удаление дубликатов
    df_cleaned = df_cleaned.drop_duplicates()

    # Заполнение пропущенных значений
    for column in df_cleaned.columns:
        if df_cleaned[column].dtype == "object":
            # Заполняем пропуски модой для категориальных данных
            df_cleaned[column] = df_cleaned[column].fillna(df_cleaned[column].mode()[0])
        else:
            # Заполняем пропуски медианой для числовых данных
            df_cleaned[column] = df_cleaned[column].fillna(df_cleaned[column].median())

    # Удаление выбросов
    for column in df_cleaned.select_dtypes(include=[np.number]):
        Q1 = df_cleaned[column].quantile(0.25)
        Q3 = df_cleaned[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_cleaned[column] = np.where(df_cleaned[column] > upper_bound, upper_bound, df_cleaned[column])
        df_cleaned[column] = np.where(df_cleaned[column] < lower_bound, lower_bound, df_cleaned[column])

    return df_cleaned


# Интерфейс приложения Streamlit
st.title("Мониторинг качества данных")
st.write("Это приложение анализирует качество данных и очищает их от ошибок и выбросов.")

# Загрузка файла
uploaded_file = st.file_uploader("Загрузите CSV файл для анализа", type="csv")
if uploaded_file is not None:
    # Чтение данных
    df = pd.read_csv(uploaded_file)
    st.write("### Загруженные данные:")
    st.write(df.head())

    # Анализ данных
    st.write("### Анализ качества данных:")
    quality_report = analyze_data_quality(df)
    for key, value in quality_report.items():
        st.write(f"**{key}:**")
        st.write(value)

    # Визуализация пропусков
    st.write("### Тепловая карта пропущенных значений:")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis", ax=ax)
    st.pyplot(fig)

    # Гистограмма пропущенных значений
    st.write("### Гистограмма пропущенных значений:")
    fig, ax = plt.subplots()
    df.isnull().sum().plot(kind='bar', ax=ax)
    ax.set_title("Пропущенные значения по колонкам")
    ax.set_ylabel("Количество пропусков")
    st.pyplot(fig)

    # Визуализация распределений
    st.write("### Гистограммы числовых данных:")
    for column in df.select_dtypes(include=[np.number]).columns:
        fig, ax = plt.subplots()
        sns.histplot(df[column], kde=True, ax=ax, color="blue")
        ax.set_title(f"Распределение: {column}")
        st.pyplot(fig)

    # Диаграммы размаха (boxplot)
    st.write("### Диаграммы размаха для числовых данных:")
    for column in df.select_dtypes(include=[np.number]).columns:
        fig, ax = plt.subplots()
        sns.boxplot(x=df[column], ax=ax)
        ax.set_title(f"Диаграмма размаха: {column}")
        st.pyplot(fig)

    # Попарные графики (pairplot)
    st.write("### Попарные графики для числовых данных:")
    if len(df.select_dtypes(include=[np.number]).columns) > 1:
        pairplot_fig = sns.pairplot(df.select_dtypes(include=[np.number]))
        st.pyplot(pairplot_fig)

    # Круговые диаграммы для категориальных данных
    st.write("### Круговые диаграммы для категориальных данных:")
    for column in df.select_dtypes(include=["object"]).columns:
        fig, ax = plt.subplots()
        df[column].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
        ax.set_title(f"Распределение: {column}")
        ax.set_ylabel("")
        st.pyplot(fig)

    # Очистка данных
    st.write("### Очистка данных")
    df_cleaned = clean_data(df)
    st.write("### Очищенные данные (первые 5 строк):")
    st.write(df_cleaned.head())

    # Подготовка данных для отчёта
    rows = []
    for key, value in quality_report.items():
        if isinstance(value, dict):  # Если значение — словарь, разворачиваем
            for sub_key, sub_value in value.items():
                rows.append({"Метрика": f"{key}_{sub_key}", "Значение": sub_value})
        else:  # Если значение — число, добавляем напрямую
            rows.append({"Метрика": key, "Значение": value})

    # Скачивание отчёта о качестве данных
    st.write("### Скачивание отчёта о качестве данных:")



    # Создание DataFrame из списка строк
    report_df = pd.DataFrame(rows)

    # Генерация CSV
    quality_csv = report_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Скачать отчёт о качестве данных",
        data=quality_csv,
        file_name="data_quality_report.csv",
    )
