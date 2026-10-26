# Used-Car Intelligence System

A machine-learning web application for used-car price estimation and purchase-quality classification, with additional tools for buyers and sellers.

## Project Overview

The system uses historical used-car data to predict an estimated asking price and classify purchase quality as Good, Average, or Poor. It also provides tools to help users explore price changes, inspect cars, and manage marketplace listings.

## Machine Learning Models

- **Random Forest Regressor:** estimates a used car's asking price.
- **Balanced Random Forest Classifier:** predicts purchase quality as Good, Average, or Poor.

The dataset cleaning and model training/comparison notebook is included in the repository. Refer to the notebook for preprocessing, experiments, and actual evaluation results. Model performance metrics should be reported only from verified notebook outputs.

## Features

- Used-car asking-price prediction
- Purchase-quality classification
- What-If Price Simulator
- Buyer Safety and Inspection Assistant
- Seller Listing Assistant
- Marketplace listing creation and management
- Search, filtering, sorting, and favourites
- Buyer enquiries and buyer–seller messaging
- User registration, login, buyer/seller roles, and password reset
- Prediction history where supported by the application

## Technology Stack

- **Language:** Python
- **Backend:** Flask
- **Frontend:** HTML, CSS, JavaScript, and Jinja2 templates
- **Machine Learning/Data:** pandas and scikit-learn
- **Database:** SQLite
- **ORM and migrations:** SQLAlchemy and Flask-Migrate
- **Version control:** Git and GitHub

## Main Files

- `app.py` — Flask application and routes
- `models.py` — database models
- `templates/` — HTML/Jinja2 templates
- `static/` — CSS and static assets
- `used_car_data_cleaning_and_model_training.ipynb` — dataset cleaning and model training/comparison notebook (confirm the exact filename in the repository)
- `used_car_classification_model.pkl` — saved classification model
- `used_car_regression_model.pkl` — saved regression model
- `used_cars_cleaned_final.csv` — cleaned dataset
- `requirements.txt` — Python dependencies
- `migrations/` — database migration files

## Setup and Run

1. Clone or download this repository.
2. Create and activate a Python virtual environment. Python 3.11 was used in the recorded development environment.
3. Install the dependencies:

   `pip install -r requirements.txt`

4. Apply database migrations:

   `python -m flask db upgrade`

5. Start the application using the project's configured Flask entry point, for example:

   `python app.py`

6. Open the local URL printed by Flask in your browser.

Check the current source code and configuration before running these commands, especially if your environment or database setup differs.

## Dataset and Evaluation

The project includes the original dataset file(s), a cleaned dataset, and a notebook documenting data preparation and model experiments. Refer to the notebook for the exact preprocessing steps, tested algorithms, train/test split, and evaluation metrics. No performance numbers are stated here because they should match the verified experiment results.

## Limitations

Predictions are estimates based on the available data and are not guarantees of market value, vehicle condition, or safety. The inspection checklist is a decision-support tool and does not replace an inspection by a qualified mechanic.

## Future Improvements

- Evaluate models on newer data
- Document reproducible model comparisons and metrics
- Add automated tests
- Improve deployment, monitoring, and security

## Academic Project

Developed as a college AI/ML project demonstrating supervised learning, model integration, and a web application.
