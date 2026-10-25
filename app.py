import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import math
from models import db, User, CarListing, Favourite, PredictionHistory, Inquiry
from flask_migrate import Migrate

app = Flask(__name__)

# ==========================================
# Database & Environment Configuration
# ==========================================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")
MIGRATIONS_DIR = os.path.join(BASE_DIR, "migrations")
DB_PATH = os.path.join(INSTANCE_DIR, "used_cars.db")
DEFAULT_DB_URI = f"sqlite:///{DB_PATH.replace(os.sep, '/')}"

ENV = os.environ.get("FLASK_ENV", "development").lower()
raw_secret = os.environ.get("SECRET_KEY")

if ENV == "production":
    if not raw_secret:
        raise RuntimeError(
            "CRITICAL SECURITY ERROR: SECRET_KEY environment variable "
            "must be explicitly set when running in production mode!"
        )
    app.config["SECRET_KEY"] = raw_secret
else:
    app.config["SECRET_KEY"] = raw_secret or "dev-insecure-secret-key-change-in-production"

app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
os.makedirs(INSTANCE_DIR, exist_ok=True)
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
    "DATABASE_URL", DEFAULT_DB_URI
)

db.init_app(app)
migrate = Migrate(app, db, directory=MIGRATIONS_DIR)


# ==========================================
# Load trained ML models
# ==========================================

classification_model = joblib.load(
    "used_car_classification_model.pkl"
)

regression_model = joblib.load(
    "used_car_regression_model.pkl"
)


# ==========================================
# Load and cache Brand & Model mapping from CSV
# ==========================================

def load_brand_models_mapping():
    try:
        df = pd.read_csv("used_cars_cleaned_final.csv")
        mapping = {}
        for brand_name, group in df.groupby("Brand"):
            if pd.isna(brand_name):
                continue
            brand_clean = str(brand_name).strip()
            if not brand_clean:
                continue
            valid_models = sorted(list({
                str(m).strip()
                for m in group["model"].dropna()
                if str(m).strip()
            }))
            if valid_models:
                mapping[brand_clean] = valid_models
        return mapping
    except Exception as e:
        print("Error loading brand models mapping:", e)
        return {}

BRAND_MODELS = load_brand_models_mapping()
ALL_BRANDS = sorted(list(BRAND_MODELS.keys()))


@app.context_processor
def inject_brands_and_models():
    return {
        "brands": ALL_BRANDS,
        "brand_models": BRAND_MODELS
    }


# ==========================================
# Home Page
# ==========================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        page="home"
    )


# ==========================================
# Buyer Mode
# ==========================================

@app.route("/buyer")
def buyer():

    return render_template(
        "index.html",
        page="buyer",
        mode="buyer"
    )


# ==========================================
# Buyer Inspection Assistant
# ==========================================

@app.route("/inspection")
def inspection():

    return render_template(
        "index.html",
        page="inspection"
    )


# ==========================================
# Compare Page
# ==========================================

@app.route("/compare")
def compare():

    return render_template(
        "index.html",
        page="compare"
    )


# ==========================================
# Seller Mode
# ==========================================

@app.route("/seller")
def seller():

    return render_template(
        "index.html",
        page="seller",
        mode="seller"
    )


# ==========================================
# Seller Listing Assistant
# ==========================================

@app.route("/listing")
def listing():

    return render_template(
        "index.html",
        page="listing",
        brand=request.args.get("brand", "").strip(),
        model=request.args.get("model", "").strip(),
        year=request.args.get("year", "").strip(),
        kmDriven=request.args.get("kmDriven", "").strip(),
        transmission=request.args.get("transmission", "").strip(),
        owner=request.args.get("owner", "").strip(),
        fuelType=request.args.get("fuelType", "").strip(),
        askingPrice=request.args.get("askingPrice", "").strip(),
        predicted_price=request.args.get("price", "").strip(),
        price_advice=request.args.get("price_advice", "").strip(),
        quality=request.args.get("quality", "").strip(),
    )


# ==========================================
# Single Car Prediction
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    # --------------------------------------
    # Get form values
    # --------------------------------------

    mode = request.form.get(
        "mode", "seller"
    ).strip().lower()

    if mode != "buyer":
        mode = "seller"

    brand = request.form.get(
        "brand", ""
    ).strip()

    model = request.form.get(
        "model", ""
    ).strip()

    year_text = request.form.get(
        "year", ""
    ).strip()

    km_text = request.form.get(
        "kmDriven", ""
    ).strip()

    transmission = request.form.get(
        "transmission", ""
    )

    owner = request.form.get(
        "owner", ""
    )

    fuel_type = request.form.get(
        "fuelType", ""
    )

    asking_price_text = request.form.get(
    "askingPrice", ""
    ).strip()


    # --------------------------------------
    # Keep entered values
    # --------------------------------------

    form_data = {

        "mode": mode,
        "brand": brand,
        "model": model,
        "year": year_text,
        "kmDriven": km_text,
        "transmission": transmission,
        "owner": owner,
        "fuelType": fuel_type,
        "askingPrice": asking_price_text

    }


    try:

        # ----------------------------------
        # Basic validation
        # ----------------------------------

        if not brand or not model:

            return render_template(
                "index.html",
                page=mode,
                error="Please select both brand and model.",
                **form_data
            )

        if brand not in BRAND_MODELS or model not in BRAND_MODELS.get(brand, []):

            return render_template(
                "index.html",
                page=mode,
                error="Please select a valid brand and model combination.",
                **form_data
            )


        if not year_text:

            return render_template(
                "index.html",
                page=mode,
                error="Please enter the manufacturing year.",
                **form_data
            )


        if not km_text:

            return render_template(
                "index.html",
                page=mode,
                error="Please enter kilometers driven.",
                **form_data
            )

        if not asking_price_text:

            return render_template(
                "index.html",
                page=mode,
                error="Please enter the seller's asking price.",
                **form_data
            )

        # ----------------------------------
        # Validate asking price
        # ----------------------------------

        try:
            asking_price = float(asking_price_text)

        except ValueError:
            return render_template(
                "index.html",
                page=mode,
                error="Please enter a valid asking price.",
                **form_data
            )

        if not math.isfinite(asking_price) or asking_price <= 0:

            return render_template(
                "index.html",
                page=mode,
                error="Asking price must be greater than zero.",
                **form_data
            )

        # ----------------------------------
        # Convert numbers
        # ----------------------------------

        year = int(year_text)

        km_driven = float(km_text)


        # ----------------------------------
        # Year validation
        # ----------------------------------

        if year < 1980 or year > 2024:

            return render_template(
                "index.html",
                page=mode,
                error=(
                    "Please enter a manufacturing year "
                    "between 1980 and 2024."
                ),
                **form_data
            )


        # ----------------------------------
        # KM validation
        # ----------------------------------

        if (
            not math.isfinite(km_driven)
            or km_driven <= 0
        ):

            return render_template(
                "index.html",
                page=mode,
                error=(
                    "Kilometers driven must be "
                    "greater than 0."
                ),
                **form_data
            )


        # ----------------------------------
        # Dropdown validation
        # ----------------------------------

        if not transmission:

            return render_template(
                "index.html",
                page=mode,
                error="Please select a transmission type.",
                **form_data
            )


        if not owner:

            return render_template(
                "index.html",
                page=mode,
                error="Please select an owner type.",
                **form_data
            )


        if not fuel_type:

            return render_template(
                "index.html",
                page=mode,
                error="Please select a fuel type.",
                **form_data
            )


        # ----------------------------------
        # Age calculation
        # ----------------------------------

        age = 2024 - year


        # ----------------------------------
        # Create input DataFrame
        # ----------------------------------

        input_data = pd.DataFrame({

            "Brand": [brand],

            "model": [model],

            "Transmission": [transmission],

            "Owner": [owner],

            "FuelType": [fuel_type],

            "Year": [year],

            "Age": [age],

            "kmDriven": [km_driven]

        })


        # ----------------------------------
        # Classification
        # ----------------------------------

        quality = classification_model.predict(
            input_data
        )[0]

        quality = str(quality)


        # ----------------------------------
        # Regression
        # ----------------------------------

        predicted_price = regression_model.predict(
            input_data
        )[0]

        predicted_price = max(
            0,
            float(predicted_price)
        )

        # Smart Price Advisor
        price_difference = asking_price - predicted_price

        if predicted_price > 0:
            difference_percent = (
            price_difference / predicted_price
        ) * 100
        else:
            difference_percent = 0

        
        if predicted_price <= 0:
            price_advice = (
                "A reliable price comparison is unavailable. "
                "Please verify the vehicle details."
            )
            price_advice_class = "average"

        elif difference_percent > 5:
            price_advice = (
                f"The asking price is {difference_percent:.2f}% "
                "above the AI estimate. Consider negotiating "
                "and checking similar listings before deciding."
            )
            price_advice_class = "poor"

        elif difference_percent < -5:
            price_advice = (
                f"The asking price is {abs(difference_percent):.2f}% "
                "below the AI estimate. Check the vehicle condition, "
                "documents and service history carefully."
            )
            price_advice_class = "good"

        else:
            price_advice = (
                f"The asking price is within 5% of the AI estimate "
                f"({abs(difference_percent):.2f}% difference). "
                "Compare similar listings and inspect the vehicle "
                "before deciding."
            )
            price_advice_class = "average"



        # ----------------------------------
        # Quality message
        # ----------------------------------

        if quality.lower() == "good":

            quality_message = (
                "This car shows favorable characteristics "
                "based on the trained machine learning model."
            )

            quality_class = "good"


        elif quality.lower() == "average":

            quality_message = (
                "This car falls in the moderate range "
                "according to the trained machine learning model."
            )

            quality_class = "average"


        else:

            quality_message = (
                "Consider checking the vehicle condition "
                "and pricing carefully."
            )

            quality_class = "poor"


        # ----------------------------------
        # Return result
        # ----------------------------------

        return render_template(

            "index.html",

            page=mode,
                mode=mode,

            prediction=quality,

            price=round(predicted_price),

            quality_message=quality_message,

            quality_class=quality_class,

            brand=brand,

            model=model,

            year=year,

            kmDriven=km_driven,

            transmission=transmission,

            owner=owner,

            fuelType=fuel_type,

            age=age,

            askingPrice=asking_price,
            
            price_difference=round(price_difference),
            
            difference_percent=round(difference_percent, 2),
            
            price_advice=price_advice,
            
            price_advice_class=price_advice_class,

    )


    except ValueError:

        return render_template(

            "index.html",

            page=mode,
                mode=mode,

            error=(
                "Please enter valid numerical values "
                "for year and kilometers driven."
            ),

            **form_data

        )


    except Exception as e:

        print(
            "Prediction Error:",
            e
        )

        return render_template(

            "index.html",

            page=mode,
                mode=mode,

            error=(
                "Something went wrong while processing "
                "the prediction. Please check your inputs."
            ),

            **form_data

        )


# ==========================================
# WHAT-IF PRICE SIMULATOR
# ==========================================

@app.route(
    "/simulate-price",
    methods=["POST"]
)
def simulate_price():
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                "success": False,
                "error": "No simulation data received."
            }), 400

        brand = str(data.get("brand", "")).strip()
        model = str(data.get("model", "")).strip()
        year_raw = data.get("year")
        km_raw = data.get("kmDriven")
        transmission = str(data.get("transmission", "")).strip()
        owner = str(data.get("owner", "First Owner")).strip()
        fuel_type = str(data.get("fuelType", "Petrol")).strip()
        baseline_price_raw = data.get("baselinePrice")

        if not brand or not model:
            return jsonify({
                "success": False,
                "error": "Brand and model are required for simulation."
            }), 400

        if brand not in BRAND_MODELS or model not in BRAND_MODELS.get(brand, []):
            return jsonify({
                "success": False,
                "error": "Invalid brand and model combination."
            }), 400

        try:
            year = int(year_raw)
        except (ValueError, TypeError):
            return jsonify({
                "success": False,
                "error": "Please provide a valid manufacturing year."
            }), 400

        if year < 1995 or year > 2024:
            return jsonify({
                "success": False,
                "error": "Manufacturing year must be between 1995 and 2024."
            }), 400

        try:
            km_driven = float(km_raw)
        except (ValueError, TypeError):
            return jsonify({
                "success": False,
                "error": "Please provide valid kilometers driven."
            }), 400

        if not math.isfinite(km_driven) or km_driven < 1000 or km_driven > 300000:
            return jsonify({
                "success": False,
                "error": "Kilometers driven must be between 1,000 and 300,000 km."
            }), 400

        if transmission not in ["Manual", "Automatic"]:
            return jsonify({
                "success": False,
                "error": "Transmission must be either Manual or Automatic."
            }), 400

        try:
            baseline_price = float(baseline_price_raw)
        except (ValueError, TypeError):
            return jsonify({
                "success": False,
                "error": "Valid baseline price is required for comparison."
            }), 400

        if not math.isfinite(baseline_price) or baseline_price <= 0:
            return jsonify({
                "success": False,
                "error": "Baseline price must be greater than zero."
            }), 400

        # Always recompute Age = 2024 - year
        age = 2024 - year

        input_data = pd.DataFrame({
            "Brand": [brand],
            "model": [model],
            "Transmission": [transmission],
            "Owner": [owner],
            "FuelType": [fuel_type],
            "Year": [year],
            "Age": [age],
            "kmDriven": [km_driven]
        })

        raw_price = regression_model.predict(input_data)[0]
        simulated_price = round(max(0, float(raw_price)))
        baseline_price_round = round(baseline_price)
        price_diff = simulated_price - baseline_price_round
        percent_diff = round((price_diff / baseline_price_round) * 100, 2) if baseline_price_round > 0 else 0.0

        return jsonify({
            "success": True,
            "simulatedPrice": simulated_price,
            "baselinePrice": baseline_price_round,
            "priceDiff": price_diff,
            "percentDiff": percent_diff,
            "simulatedYear": year,
            "simulatedAge": age,
            "simulatedKm": round(km_driven),
            "simulatedTransmission": transmission
        })

    except Exception as e:
        print("Simulation Error:", e)
        return jsonify({
            "success": False,
            "error": "An error occurred while simulating the price. Please verify your inputs."
        }), 500


# ==========================================
# COMPARE TWO CARS
# ==========================================

@app.route(
    "/compare-cars",
    methods=["POST"]
)
def compare_cars():

    try:

        # ----------------------------------
        # Get JSON data from JavaScript
        # ----------------------------------

        data = request.get_json()


        if not data:

            return jsonify({

                "success": False,

                "error": "No car data received."

            })


        car_a = data.get(
            "carA"
        )

        car_b = data.get(
            "carB"
        )


        if not car_a or not car_b:

            return jsonify({

                "success": False,

                "error": "Details for both cars are required."

            })


        # ----------------------------------
        # Function for validating one car
        # ----------------------------------

        def validate_car(car):

            required_fields = [

                "brand",
                "model",
                "year",
                "km",
                "transmission",
                "owner",
                "fuel"

            ]


            for field in required_fields:

                if not car.get(field):

                    return (
                        False,
                        "Please fill all details for both cars."
                    )

            brand_val = str(car.get("brand", "")).strip()
            model_val = str(car.get("model", "")).strip()

            if brand_val not in BRAND_MODELS or model_val not in BRAND_MODELS.get(brand_val, []):

                return (
                    False,
                    f"Please select a valid brand and model combination for {brand_val or 'car'}."
                )


            year = int(
                car["year"]
            )

            km = float(
                car["km"]
            )


            if year < 1980 or year > 2024:

                return (

                    False,

                    "Manufacturing year must be "
                    "between 1980 and 2024."

                )


            if (
                not math.isfinite(km)
                or km <= 0
            ):

                return (

                    False,

                    "Kilometers driven must be "
                    "greater than 0."

                )


            return True, ""


        # ----------------------------------
        # Validate Car A
        # ----------------------------------

        valid_a, error_a = validate_car(
            car_a
        )


        if not valid_a:

            return jsonify({

                "success": False,

                "error": error_a

            })


        # ----------------------------------
        # Validate Car B
        # ----------------------------------

        valid_b, error_b = validate_car(
            car_b
        )


        if not valid_b:

            return jsonify({

                "success": False,

                "error": error_b

            })


        # ----------------------------------
        # Predict one car
        # ----------------------------------

        def predict_car(car):

            brand = car["brand"]

            model = car["model"]

            year = int(
                car["year"]
            )

            km = float(
                car["km"]
            )

            transmission = car[
                "transmission"
            ]

            owner = car[
                "owner"
            ]

            fuel = car[
                "fuel"
            ]


            # Same Age calculation
            # used during model training

            age = 2024 - year


            # Create DataFrame

            input_data = pd.DataFrame({

                "Brand": [brand],

                "model": [model],

                "Transmission": [
                    transmission
                ],

                "Owner": [
                    owner
                ],

                "FuelType": [
                    fuel
                ],

                "Year": [year],

                "Age": [age],

                "kmDriven": [km]

            })


            # Classification

            quality = classification_model.predict(
                input_data
            )[0]

            quality = str(
                quality
            )


            # Regression

            price = regression_model.predict(
                input_data
            )[0]


            price = max(
                0,
                float(price)
            )


            return {

                "brand": brand,

                "model": model,

                "year": year,

                "km": km,

                "quality": quality,

                "price": round(price)

            }


        # ----------------------------------
        # Predict both cars
        # ----------------------------------

        result_a = predict_car(
            car_a
        )

        result_b = predict_car(
            car_b
        )


        # ----------------------------------
        # AI Recommendation
        # ----------------------------------

        quality_rank = {
            "poor": 1,
            "average": 2,
            "good": 3
        }

        quality_a = quality_rank.get(
            result_a["quality"].lower(), 0
        )

        quality_b = quality_rank.get(
            result_b["quality"].lower(), 0
        )

        price_a = result_a["price"]
        price_b = result_b["price"]

        price_difference = abs(price_a - price_b)

        # Prioritize purchase quality
        if quality_a > quality_b:

            recommendation = (
                f"Recommended: Car A "
                f"({result_a['brand']} {result_a['model']}). "
                f"It received a better AI purchase-quality "
                f"classification than Car B."
            )

        elif quality_b > quality_a:

            recommendation = (
                f"Recommended: Car B "
                f"({result_b['brand']} {result_b['model']}). "
                f"It received a better AI purchase-quality "
                f"classification than Car A."
            )

        # Same quality: compare estimated prices carefully
        elif price_difference < 10000:

            recommendation = (
                "Both cars have the same AI purchase quality "
                "and similar estimated values. Compare their "
                "condition, service history and inspection "
                "results before deciding."
            )

        else:

            recommendation = (
                f"Both cars have the same AI purchase quality. "
                f"Car A is estimated at ₹{price_a:,.0f}, "
                f"while Car B is estimated at ₹{price_b:,.0f}. "
                f"The higher estimated value does not necessarily "
                f"mean that car is the better purchase. Check "
                f"condition, maintenance and market price."
            )


        # ----------------------------------
        # Return JSON
        # ----------------------------------

        return jsonify({

            "success": True,

            "carA": result_a,

            "carB": result_b,

            "recommendation": recommendation

        })


    except ValueError:

        return jsonify({

            "success": False,

            "error": (
                "Please enter valid numerical values "
                "for year and kilometers."
            )

        })


    except Exception as e:

        print(
            "Comparison Error:",
            e
        )

        return jsonify({

            "success": False,

            "error": (
                "Something went wrong while comparing "
                "the cars. Please try again."
            )

        })


# ==========================================
# Run Flask Application
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True
    )