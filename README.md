# DreamHouse AI

DreamHouse AI — це Machine Learning пет-проєкт (Full-Stack), який прогнозує орієнтовну вартість будівництва будинку на основі його параметрів (площа, матеріали, наявність басейну тощо).

Проєкт демонструє повний цикл розробки ML-продукту: від генерації даних та тренування моделі до розгортання API та створення клієнтського веб-інтерфейсу.

## Технологічний стек

- Machine Learning: Python, Pandas, Scikit-Learn, XGBoost

- Backend: FastAPI, Pydantic, Uvicorn

- Frontend: HTML5, JavaScript (Fetch API), Tailwind CSS

- Аналітика: Jupyter Notebook, Matplotlib, Seaborn

## Структура проєкту

DreamHouse_AI/
│
├── data_generation/           
│   └── generate_houses.py     # Скрипт генерації синтетичного датасету
│
├── data/                      
│   └── raw/
│       └── houses_dataset.csv # Згенеровані дані (ігнорується git)
│
├── notebooks/                 
│   ├── 01_eda.ipynb           # Exploratory Data Analysis (графіки та аналіз)
│   └── 02_model_train.ipynb   # Навчання моделі XGBoost та її збереження
│
├── models/                    
│   ├── house_pricer_xgb.pkl   # Натренована модель
│   └── model_features.pkl     # Збережений список колонок (для бекенду)
│
├── backend/                   
│   └── main.py                # FastAPI сервер
│
└── frontend/                  
    └── index.html             # Веб-інтерфейс користувача


## Як запустити проєкт локально

1. Підготовка середовища

Переконайтеся, що у вас встановлено Python 3.9+.
Створіть віртуальне середовище та активуйте його:
```bash
python3 -m venv .venv
source .venv/bin/activate  # Для macOS/Linux
.venv\Scripts\activate   # Для Windows
```

Встановіть необхідні бібліотеки:
``` bash
python3 -m pip install fastapi uvicorn pydantic joblib pandas numpy scikit-learn xgboost matplotlib seaborn jupyter ipykernel
```

2. Генерація даних

Оскільки проєкт використовує синтетичні дані, спочатку згенеруйте їх:
```bash
cd data_generation
python generate_houses.py
```

Це створить файл data/raw/houses_dataset.csv.

3. Навчання моделі

Відкрийте файл notebooks/02_model_train.ipynb у вашому редакторі (наприклад, VS Code).

Запустіть усі клітинки.

В останній клітинці модель буде збережена у папку models/.

4. Запуск Backend-сервера

Перейдіть у папку бекенду та запустіть сервер Uvicorn:
``` bash
cd backend
uvicorn main:app --reload
```

API буде доступне за адресою: http://127.0.0.1:8000
Swagger UI доступна за адресою: http://127.0.0.1:8000/docs

5. Запуск Frontend-інтерфейсу

Для фронтенду не потрібен окремий сервер. Просто відкрийте файл frontend/index.html у будь-якому сучасному браузері:

macOS: open frontend/index.html

Windows: start frontend/index.html

Або просто перетягніть файл у вікно браузера.

Розроблено як демонстраційний проєкт для портфоліо ML-інженера.