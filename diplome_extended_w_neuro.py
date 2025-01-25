import tensorflow as tf
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Настройки страницы
st.set_page_config(
    page_title="Нейросетевой анализатор",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Инициализация состояния сессии
def init_session():
    session_defaults = {
        'model': None,
        'scaler': None,
        'encoders': None,
        'train_data': None,
        'target_col': None,
        'history': None,
        'feature_columns': None
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Функция предобработки данных
def preprocess_data(df, target_column):
    try:
        if target_column not in df.columns:
            raise ValueError("Целевая переменная отсутствует в данных.")

        df_clean = df.copy()
        y = df_clean.pop(target_column)
        X = df_clean

        # Кодирование категориальных признаков
        encoders = {}
        for col in X.select_dtypes(include=['object']).columns:
            encoders[col] = {v: k for k, v in enumerate(X[col].unique())}
            X[col] = X[col].map(encoders[col])

        # Кодирование целевой переменной
        target_encoder = {v: k for k, v in enumerate(y.unique())}
        y = y.map(target_encoder)

        return X.astype('float32'), y.astype('float32'), encoders, target_encoder

    except Exception as e:
        st.error(f"Ошибка предобработки: {str(e)}")
        raise

# Создание модели
def create_model(input_shape, n_classes, params):
    try:
        model = Sequential()

        # Добавление скрытых слоёв
        for _ in range(params['hidden_layers']):
            model.add(Dense(
                params['units'],
                activation=params['activation'],
                kernel_initializer='he_normal'
            ))
            if params['dropout'] > 0:
                model.add(Dropout(params['dropout']))

        # Выходной слой
        if n_classes == 2:
            model.add(Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:
            model.add(Dense(n_classes, activation='softmax'))
            loss = 'sparse_categorical_crossentropy'

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=params['lr']),
            loss=loss,
            metrics=['accuracy']
        )
        return model

    except Exception as e:
        st.error(f"Ошибка создания модели: {str(e)}")
        raise

# Основное приложение
def main():
    init_session()

    st.title("🧠 Интерактивный нейросетевой анализатор")
    st.markdown("---")

    # Сайдбар с настройками
    with st.sidebar:
        st.header("⚙️ Параметры модели")

        params = {
            'hidden_layers': st.slider("Количество скрытых слоёв", 1, 5, 2),
            'units': st.slider("Количество нейронов в слое", 16, 256, 64),
            'activation': st.selectbox("Функция активации", ['relu', 'elu', 'tanh']),
            'dropout': st.slider("Dropout", 0.0, 0.5, 0.2),
            'lr': st.slider("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f"),
            'epochs': st.slider("Количество эпох", 10, 200, 50),
            'batch_size': st.selectbox("Размер батча", [32, 64, 128, 256])
        }

    # Основные вкладки
    tab1, tab2, tab3 = st.tabs(["📁 Данные", "🎓 Обучение", "🔮 Прогнозирование"])

    # Вкладка данных
    with tab1:
        st.header("Загрузка и анализ данных")

        uploaded_file = st.file_uploader("Загрузите CSV файл", type="csv")

        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.train_data = df

                st.success(f"✅ Успешно загружено {len(df)} записей")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Первые 5 строк")
                    st.dataframe(df.head())

                with col2:
                    st.subheader("Информация о данных")
                    st.write(f"**Количество признаков:** {df.shape[1]}")
                    st.write(f"**Типы данных:**")
                    st.json(df.dtypes.astype(str).to_dict())

                    if st.button("Анализ пропущенных значений"):
                        nulls = df.isnull().sum()
                        fig, ax = plt.subplots()
                        nulls[nulls > 0].plot(kind='bar', ax=ax)
                        ax.set_title("Пропущенные значения")
                        st.pyplot(fig)

            except Exception as e:
                st.error(f"Ошибка загрузки данных: {str(e)}")

    # Вкладка обучения
    with tab2:
        st.header("Обучение модели")

        if st.session_state.train_data is not None:
            df = st.session_state.train_data
            target_col = st.selectbox("Выберите целевую переменную", df.columns)

            if st.button("Запустить обучение", type="primary"):
                with st.spinner("Обучение модели..."):
                    try:
                        # Предобработка данных
                        X, y, encoders, target_encoder = preprocess_data(df, target_col)

                        # Сохранение состояния
                        st.session_state.encoders = {
                            'features': encoders,
                            'target': target_encoder
                        }

                        # Масштабирование
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        st.session_state.scaler = scaler
                        st.session_state.feature_columns = X.columns.tolist()

                        # Создание и обучение модели
                        n_classes = len(target_encoder)
                        model = create_model(X_scaled.shape[1], n_classes, params)

                        history = model.fit(
                            X_scaled, y,
                            epochs=params['epochs'],
                            batch_size=params['batch_size'],
                            validation_split=0.2,
                            verbose=0
                        )

                        st.session_state.model = model
                        st.session_state.history = history

                        # Визуализация результатов
                        st.subheader("Результаты обучения")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric("Финальная точность", f"{history.history['accuracy'][-1]:.2%}")
                            st.metric("Лучшая точность валидации", f"{max(history.history['val_accuracy']):.2%}")

                        with col2:
                            fig = plt.figure(figsize=(10, 4))
                            plt.plot(history.history['loss'], label='Train Loss')
                            plt.plot(history.history['val_loss'], label='Validation Loss')
                            plt.title('График потерь')
                            plt.legend()
                            st.pyplot(fig)

                        st.success("Обучение завершено успешно!")

                    except Exception as e:
                        st.error(f"Ошибка: {str(e)}")
        else:
            st.warning("Сначала загрузите данные на вкладке 📁 Данные")

    # Вкладка прогнозирования
    with tab3:
        st.header("Прогнозирование на новых данных")

        if st.session_state.model:
            predict_file = st.file_uploader("Загрузите данные для прогноза", type="csv")

            if predict_file:
                try:
                    df_pred = pd.read_csv(predict_file)

                    if st.button("Выполнить прогноз"):
                        with st.spinner("Анализ данных..."):
                            # Предобработка
                            df_processed = df_pred.copy()

                            if st.session_state.target_col and st.session_state.target_col in df_processed.columns:
                                df_processed = df_processed.drop(columns=[st.session_state.target_col])

                            # Проверка совпадения столбцов
                            expected_columns = set(st.session_state.feature_columns)
                            missing_cols = expected_columns - set(df_processed.columns)
                            extra_cols = set(df_processed.columns) - expected_columns

                            if missing_cols:
                                st.error(f"Отсутствуют необходимые столбцы: {', '.join(missing_cols)}")
                                return

                            if extra_cols:
                                st.warning(f"Игнорируем неожиданные столбцы: {', '.join(extra_cols)}")
                                df_processed = df_processed[list(expected_columns)]

                            for col, encoder in st.session_state.encoders['features'].items():
                                df_processed[col] = df_processed[col].apply(lambda x: encoder.get(x, -1))

                            non_numeric = df_processed.select_dtypes(exclude=[np.number]).columns
                            if not non_numeric.empty:
                                st.error(f"Обнаружены нечисловые данные в столбцах: {', '.join(non_numeric)}")
                                return

                            # Приведение порядка столбцов к ожидаемому
                            df_processed = df_processed[st.session_state.feature_columns]

                            # Масштабирование
                            X_pred = df_processed.astype('float32').values
                            X_scaled = st.session_state.scaler.transform(X_pred)

                            # Прогнозирование
                            predictions = st.session_state.model.predict(X_scaled)
                            inv_target = {v: k for k, v in st.session_state.encoders['target'].items()}

                            if predictions.shape[1] == 1:
                                results = [inv_target.get(int(round(p[0])), 'UNKNOWN') for p in predictions]
                                probs = [p[0] for p in predictions]
                            else:
                                results = [inv_target.get(np.argmax(p), 'UNKNOWN') for p in predictions]
                                probs = [np.max(p) for p in predictions]

                            result_df = pd.DataFrame({
                                'Прогноз': results,
                                'Вероятность': probs
                            }).style.format({'Вероятность': '{:.2%}'})

                            st.subheader("Результаты прогнозирования")
                            st.dataframe(result_df, height=400)

                except Exception as e:
                    st.error(f"Ошибка прогнозирования: {str(e)}")
                    st.error(f"Проблемные данные:\n{df_pred.dtypes}\nПример строки:\n{df_pred.iloc[0]}")
        else:
            st.warning("Сначала обучите модель на вкладке 🎓 Обучение")

if __name__ == "__main__":
    main()
