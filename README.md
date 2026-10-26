Weather Prediction API
    A lightweight, machine learning-powered web application and REST API built using FastAPI and Scikit-learn to predict whether rain is expected based on real-time weather parameters (temperature, humidity, and wind speed).

Features
    FastAPI Backend: High-performance, automatic interactive API documentation (/docs).
    Machine Learning Model: Trained using a Decision Tree Classifier (criterion='entropy') via Scikit-learn.
    Data Validation: Uses Pydantic to validate user inputs securely and effectively within specific ranges.
    Interactive UI: Includes both a home endpoint and automated Swagger UI testing.

Tech Stack
    Python (Core Language)
    FastAPI (Web Framework)
    Uvicorn (ASGI Server)
    Scikit-learn (Machine Learning Model)
    Pandas (Data Handling)
    Pydantic (Data Validation)