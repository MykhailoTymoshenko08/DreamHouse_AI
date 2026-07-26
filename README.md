python3 -m pip install jupyter ipykernel pandas numpy scikit-learn xgboost


DreamHouse_AI/
│
├── data/                   <- 💾 Папка для даних (ЇЇ НЕ ДОДАЮТЬ У GIT!)
│   ├── raw/                <- Сирі дані (те, що згенерував наш скрипт)
│   └── processed/          <- Очищені та підготовлені дані
│
├── data_generation/        <- 🏭 Скрипти для збору/генерації даних
│   └── generate_houses.py  <- Наш перший скрипт!
│
├── notebooks/              <- 📓 Лабораторія (Jupyter Notebooks)
│   ├── 01_eda.ipynb        <- Тут ми будемо малювати графіки і дивитися на дані
│   └── 02_model_train.ipynb<- Тут експериментуємо з XGBoost
│
├── models/                 <- 🧠 Збережені "мізки"
│   └── house_pricer.onnx   <- Наша готова модель
│
├── backend/                <- ⚙️ Традиційний бекенд (FastAPI)
│   └── main.py             <- API, яке приймає JSON і видає ціну
│
├── frontend/               <- 🖥️ Веб-інтерфейс (зробимо на Streamlit або HTML/JS)
│   └── app.py              <- Користувацький інтерфейс з повзунками і картинками
│
├── requirements.txt        <- Список бібліотек (pandas, xgboost, fastapi тощо)
└── README.md               <- Опис проєкту для рекрутерів
