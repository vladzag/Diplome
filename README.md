# Data Quality Monitor 🛠️📊

**Data Quality Monitor** — это интерактивное веб-приложение для анализа и очистки данных, построенное на базе Streamlit. Приложение автоматически выявляет аномалии, пропуски, дубликаты и визуализирует данные, а также предоставляет инструменты для их очистки.

---

## 📋 Основные функции

1. **Анализ данных**:
   - Пропущенные значения.
   - Уникальные значения.
   - Дубликаты.
   - Выбросы (через межквартильный размах).

2. **Визуализация**:
   - Тепловая карта пропущенных значений.
   - Гистограммы распределений.
   - Диаграммы размаха (boxplot).
   - Попарные графики для числовых данных.
   - Круговые диаграммы для категориальных данных.

3. **Очистка данных**:
   - Удаление дубликатов.
   - Заполнение пропусков (медиана для чисел, мода для категорий).
   - Коррекция выбросов.

4. **Отчёт**:
   - Генерация CSV-отчёта с метриками качества данных.

---

## 🚀 Установка и запуск

1. **Клонируйте репозиторий**:
   ```bash
   git clone https://github.com/vladzag/Diplome.git
   cd data-quality-monitor
2. **Установите зависимости**:

    ```bash
    pip install -r requirements.txt


3. **Запустите приложение**:

    ```bash
    streamlit run app.py

🖥️ Использование
1. Загрузите CSV-файл через интерфейс приложения.

2. Анализ данных:

    * Автоматически генерируется отчёт с метриками.

    * Визуализации отображаются в реальном времени.

3. Очистка данных:

    * Нажмите на раздел «Очистка данных», чтобы получить обработанный датасет.

4. Скачайте отчёт в формате CSV.

🛠️ Технологии


* Python 3.9+

* **Pandas** — обработка данных.

* **Streamlit** — веб-интерфейс.

* **Seaborn/Matplotlib** — визуализация.

* **NumPy** — математические операции.

📌 Особенности реализации

* Обработка выбросов: Используется метод межквартильного размаха (IQR).

* Заполнение пропусков:
  * Числовые данные: медиана. 
  * Категориальные данные: мода.

* Автоматическая генерация графиков для всех числовых и категориальных колонок.

📄 Лицензия
Проект распространяется под лицензией MIT. 

🤝 Как внести вклад

* Форкните репозиторий.

* Создайте ветку с вашим фичей (***git checkout -b feature/AmazingFeature***).

* Зафиксируйте изменения (***git commit -m 'Add some AmazingFeature'***).

* Запушьте ветку (***git push origin feature/AmazingFeature***).

* Откройте Pull Request.

Разработано с ❤️ для анализа данных.