import os
import re
import secrets
import smtplib
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash, g
import pandas as pd
import joblib
import math
from models import db, User, CarListing, Favourite, PredictionHistory, Inquiry, PasswordResetToken, InquiryMessage, utcnow
from flask_migrate import Migrate
from dotenv import load_dotenv
load_dotenv()

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

# Session and cookie security
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

def generate_csrf_token():
    if "_csrf_token" not in session:
        session["_csrf_token"] = secrets.token_hex(16)
    return session["_csrf_token"]

EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def is_valid_email(email):
    return bool(email and EMAIL_REGEX.match(email.strip()))


def send_reset_password_email(to_email, recipient_name, reset_url):
    """
    Sends a password reset email using standard library smtplib and EmailMessage.
    Uses SMTP environment variables if configured.
    Falls back to development console output ONLY if DEBUG_EMAIL_LINKS is True and ENV == 'development'.
    Never logs credentials, passwords, or links in production.
    """
    mail_server = os.environ.get("MAIL_SERVER")
    mail_port = int(os.environ.get("MAIL_PORT", 587))
    mail_use_tls = os.environ.get("MAIL_USE_TLS", "true").lower() in ("true", "1", "yes")
    mail_use_ssl = os.environ.get("MAIL_USE_SSL", "false").lower() in ("true", "1", "yes")
    mail_username = os.environ.get("MAIL_USERNAME")
    mail_password = os.environ.get("MAIL_PASSWORD")
    mail_sender = os.environ.get("MAIL_DEFAULT_SENDER", "UsedCar AI <no-reply@usedcarai.com>")
    debug_email_links = os.environ.get("DEBUG_EMAIL_LINKS", "false").lower() in ("true", "1", "yes")

    # Safe development-only terminal output
    if ENV == "development" and debug_email_links:
        print(f"[DEV DEBUG ONLY] Password reset link for {to_email}: {reset_url}")

    if not mail_server:
        return False

    try:
        msg = EmailMessage()
        msg["Subject"] = "Password Reset Request - UsedCar AI"
        msg["From"] = mail_sender
        msg["To"] = to_email

        plain_text = f"""Hello {recipient_name},

We received a request to reset the password for your UsedCar AI account.

To choose a new password, click the link below (valid for 30 minutes):
{reset_url}

If you did not request this change, please ignore this email. Your password will remain unchanged.

Best regards,
The UsedCar AI Team
"""
        html_text = f"""<!DOCTYPE html>
<html>
<body style="font-family: Arial, sans-serif; background-color: #0d131d; color: #ffffff; padding: 24px;">
    <div style="max-width: 540px; margin: 0 auto; background: #161f30; border: 1px solid rgba(255,93,93,0.3); border-radius: 12px; padding: 30px;">
        <h2 style="color: #ff5d5d; margin-top: 0;">UsedCar AI Password Reset</h2>
        <p style="color: #cbd5e1; font-size: 15px;">Hello {recipient_name},</p>
        <p style="color: #94a3b8; font-size: 14px; line-height: 1.5;">
            We received a request to reset the password for your account. Click the button below to set a new password. This link will expire in <strong>30 minutes</strong>.
        </p>
        <div style="text-align: center; margin: 28px 0;">
            <a href="{reset_url}" style="background: linear-gradient(90deg, #ff4e5b, #ff7544); color: #ffffff; text-decoration: none; padding: 12px 28px; border-radius: 8px; font-weight: bold; font-size: 15px; display: inline-block;">
                Reset Password
            </a>
        </div>
        <p style="color: #64748b; font-size: 12px; line-height: 1.4;">
            If you did not make this request, you can safely ignore this email. Your account remains secure.<br>
            Direct link: <a href="{reset_url}" style="color: #ff7575;">{reset_url}</a>
        </p>
    </div>
</body>
</html>
"""
        msg.set_content(plain_text)
        msg.add_alternative(html_text, subtype="html")

        if mail_use_ssl:
            with smtplib.SMTP_SSL(mail_server, mail_port, timeout=10) as server:
                if mail_username and mail_password:
                    server.login(mail_username, mail_password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(mail_server, mail_port, timeout=10) as server:
                if mail_use_tls:
                    server.starttls()
                if mail_username and mail_password:
                    server.login(mail_username, mail_password)
                server.send_message(msg)
        return True
    except Exception as e:
        print(f"SMTP Delivery Notice: Failed to send reset email: {e}")
        return False


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


@app.before_request
def load_logged_in_user():
    user_id = session.get("user_id")
    if user_id is None:
        g.user = None
    else:
        u = db.session.get(User, user_id)
        auth_hash = session.get("auth_hash")
        # Invalidate session if user's password was changed
        if u and auth_hash and auth_hash != u.auth_fingerprint:
            session.clear()
            g.user = None
        else:
            g.user = u


@app.context_processor
def inject_brands_and_models():
    return {
        "brands": ALL_BRANDS,
        "brand_models": BRAND_MODELS,
        "current_user": getattr(g, "user", None),
        "csrf_token": generate_csrf_token
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

    if getattr(g, "user", None) and g.user.is_seller:
        flash("You are signed in with a Seller account. Redirected to Seller Mode.", "info")
        return redirect(url_for("seller"))

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

    if getattr(g, "user", None) and g.user.is_buyer:
        flash("You are signed in with a Buyer account. Redirected to Buyer Mode.", "info")
        return redirect(url_for("buyer"))

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

    if getattr(g, "user", None) and g.user.is_buyer:
        flash("The Seller Listing Assistant is available only for Seller accounts. Redirected to Buyer Mode.", "info")
        return redirect(url_for("buyer"))

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
    # Get form values & enforce authenticated role
    # --------------------------------------

    if getattr(g, "user", None):
        if g.user.is_buyer:
            mode = "buyer"
        elif g.user.is_seller:
            mode = "seller"
        else:
            mode = request.form.get("mode", "seller").strip().lower()
    else:
        mode = request.form.get("mode", "seller").strip().lower()

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
# Authentication Routes (Register, Login, Logout)
# ==========================================

@app.route("/register", methods=["GET", "POST"])
def register():
    if getattr(g, "user", None):
        return redirect(url_for("seller" if g.user.is_seller else "buyer"))

    if request.method == "POST":
        token = request.form.get("csrf_token")
        if not token or token != session.get("_csrf_token"):
            flash("Security check failed. Please submit the form again.", "error")
            return render_template("index.html", page="register", form_data=request.form)

        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        phone = request.form.get("phone", "").strip()
        role = request.form.get("role", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        errors = []
        if not full_name or len(full_name) < 2 or len(full_name) > 100:
            errors.append("Please enter a valid full name (2 to 100 characters).")

        if not is_valid_email(email):
            errors.append("Please enter a valid email address.")
        elif User.query.filter_by(email=email).first():
            errors.append("An account with this email address already exists.")

        if role not in ("buyer", "seller"):
            errors.append("Please select a valid account type: Buyer or Seller.")

        if not password or len(password) < 8:
            errors.append("Password must be at least 8 characters long.")
        elif password != confirm_password:
            errors.append("Password and confirmation password do not match.")

        if errors:
            for error in errors:
                flash(error, "error")
            return render_template("index.html", page="register", form_data=request.form)

        try:
            new_user = User(
                full_name=full_name,
                email=email,
                phone=phone if phone else None,
                role=role
            )
            new_user.set_password(password)
            db.session.add(new_user)
            db.session.commit()

            # Automatic login upon successful registration
            session.clear()
            session["user_id"] = new_user.id
            session["auth_hash"] = new_user.auth_fingerprint
            session["_csrf_token"] = secrets.token_hex(16)
            flash(f"Welcome, {new_user.full_name}! Your {new_user.role.capitalize()} account was created.", "success")
            return redirect(url_for("seller" if new_user.is_seller else "buyer"))

        except Exception as e:
            db.session.rollback()
            print("Registration Error:", e)
            flash("An unexpected error occurred during registration. Please try again.", "error")
            return render_template("index.html", page="register", form_data=request.form)

    return render_template("index.html", page="register", form_data={})


@app.route("/login", methods=["GET", "POST"])
def login():
    if getattr(g, "user", None):
        return redirect(url_for("seller" if g.user.is_seller else "buyer"))

    next_url = request.args.get("next") or request.form.get("next")

    if request.method == "POST":
        token = request.form.get("csrf_token")
        if not token or token != session.get("_csrf_token"):
            flash("Security check failed. Please submit the form again.", "error")
            return render_template("index.html", page="login", form_data=request.form, next=next_url)

        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not email or not password:
            flash("Please enter both email address and password.", "error")
            return render_template("index.html", page="login", form_data=request.form, next=next_url)

        user = User.query.filter_by(email=email).first()
        if not user or not user.check_password(password):
            flash("Invalid email address or password. Please try again.", "error")
            return render_template("index.html", page="login", form_data={"email": email}, next=next_url)

        session.clear()
        session["user_id"] = user.id
        session["auth_hash"] = user.auth_fingerprint
        session["_csrf_token"] = secrets.token_hex(16)
        flash(f"Welcome back, {user.full_name}!", "success")

        # Validate redirect target to prevent open redirect vulnerabilities
        if next_url and next_url.startswith("/") and not next_url.startswith("//") and not next_url.startswith("\\"):
            return redirect(next_url)

        return redirect(url_for("seller" if user.is_seller else "buyer"))

    return render_template("index.html", page="login", form_data={}, next=next_url)


@app.route("/logout", methods=["GET", "POST"])
def logout():
    if request.method == "GET":
        flash("Please use the Sign Out button to log out securely.", "info")
        return redirect(url_for("home"))

    token = request.form.get("csrf_token")
    expected_token = session.get("_csrf_token")
    if not token or token != expected_token:
        flash("Invalid security token. Please try again.", "error")
        return redirect(url_for("home"))

    session.clear()
    flash("You have been logged out successfully.", "info")
    return redirect(url_for("login"))


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if getattr(g, "user", None):
        return redirect(url_for("seller" if g.user.is_seller else "buyer"))

    if request.method == "POST":
        token = request.form.get("csrf_token")
        if not token or token != session.get("_csrf_token"):
            flash("Security check failed. Please submit the form again.", "error")
            return render_template("index.html", page="forgot_password", form_data=request.form)

        email = request.form.get("email", "").strip().lower()

        # Anti-enumeration response: always display this exact message
        generic_msg = (
            "If an account exists for that email address, a password reset link "
            "has been sent. Please check your inbox and spam folder."
        )

        if is_valid_email(email):
            user = User.query.filter_by(email=email).first()
            if user:
                try:
                    # Invalidate existing unused tokens for this user
                    existing_tokens = PasswordResetToken.query.filter_by(
                        user_id=user.id,
                        used_at=None
                    ).all()
                    for t in existing_tokens:
                        t.used_at = datetime.utcnow()

                    # Generate cryptographically secure token
                    raw_token = secrets.token_urlsafe(32)
                    token_hash = PasswordResetToken.hash_token(raw_token)
                    expires_at = datetime.utcnow() + timedelta(minutes=30)

                    reset_record = PasswordResetToken(
                        user_id=user.id,
                        token_hash=token_hash,
                        expires_at=expires_at
                    )
                    db.session.add(reset_record)
                    db.session.commit()

                    # Trusted base URL from configuration (preventing Host header injection)
                    app_base_url = os.environ.get("APP_BASE_URL", "http://127.0.0.1:5000").rstrip("/")
                    reset_url = f"{app_base_url}/reset-password/{raw_token}"

                    # Send reset email
                    send_reset_password_email(user.email, user.full_name, reset_url)

                except Exception as e:
                    db.session.rollback()
                    print("Forgot Password Error:", e)

        flash(generic_msg, "info")
        return redirect(url_for("login"))

    return render_template("index.html", page="forgot_password", form_data={})


@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    if getattr(g, "user", None):
        return redirect(url_for("seller" if g.user.is_seller else "buyer"))

    raw_token = str(token).strip()
    token_hash = PasswordResetToken.hash_token(raw_token)

    # Validate token
    reset_record = PasswordResetToken.query.filter_by(token_hash=token_hash).first()

    if not reset_record or not reset_record.is_valid:
        flash(
            "This password reset link is invalid, has expired, or has already been used. "
            "Please request a new reset link.",
            "error"
        )
        return redirect(url_for("forgot_password"))

    if request.method == "POST":
        csrf_tok = request.form.get("csrf_token")
        if not csrf_tok or csrf_tok != session.get("_csrf_token"):
            flash("Security check failed. Please submit the form again.", "error")
            return render_template("index.html", page="reset_password", token=raw_token)

        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        errors = []
        if not password or len(password) < 8:
            errors.append("Password must be at least 8 characters long.")
        elif password != confirm_password:
            errors.append("Password and confirmation password do not match.")

        if errors:
            for err in errors:
                flash(err, "error")
            return render_template("index.html", page="reset_password", token=raw_token)

        # Atomic token consumption and password update
        try:
            # Re-verify record in current transaction
            if not reset_record.is_valid:
                flash("This password reset link has already been used or expired.", "error")
                return redirect(url_for("forgot_password"))

            user = reset_record.user
            user.set_password(password)
            reset_record.used_at = datetime.utcnow()

            db.session.commit()

            # Invalidate any active session
            session.clear()

            flash(
                "Your password has been successfully reset! You can now log in with your new password.",
                "success"
            )
            return redirect(url_for("login"))

        except Exception as e:
            db.session.rollback()
            print("Password Reset Error:", e)
            flash("An unexpected error occurred while resetting your password. Please try again.", "error")
            return render_template("index.html", page="reset_password", token=raw_token)

    return render_template("index.html", page="reset_password", token=raw_token)






# ==========================================
# Real Car Marketplace Routes
# ==========================================

VALID_FUEL_TYPES = ["Petrol", "Diesel", "CNG", "LPG", "Electric"]
VALID_TRANSMISSIONS = ["Manual", "Automatic"]
VALID_OWNERS = ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"]


@app.route("/marketplace")
def marketplace():
    """Browse, search, filter, and sort active car listings."""
    q = request.args.get("q", "").strip()
    brand = request.args.get("brand", "").strip()
    min_price = request.args.get("min_price", "").strip()
    max_price = request.args.get("max_price", "").strip()
    min_year = request.args.get("min_year", "").strip()
    max_year = request.args.get("max_year", "").strip()
    fuel_type = request.args.get("fuel_type", "").strip()
    transmission = request.args.get("transmission", "").strip()
    owner = request.args.get("owner", "").strip()
    sort = request.args.get("sort", "newest").strip().lower()

    query = CarListing.query.filter_by(status="active")

    # Keyword search across brand, model, title, description
    if q:
        search_pattern = f"%{q}%"
        query = query.filter(
            db.or_(
                CarListing.brand.ilike(search_pattern),
                CarListing.model.ilike(search_pattern),
                CarListing.title.ilike(search_pattern),
                CarListing.description.ilike(search_pattern)
            )
        )

    # Brand filter
    if brand and brand != "All":
        query = query.filter(CarListing.brand.ilike(brand))

    # Price range filter
    if min_price:
        try:
            query = query.filter(CarListing.asking_price >= float(min_price))
        except ValueError:
            pass
    if max_price:
        try:
            query = query.filter(CarListing.asking_price <= float(max_price))
        except ValueError:
            pass

    # Year range filter
    if min_year:
        try:
            query = query.filter(CarListing.year >= int(min_year))
        except ValueError:
            pass
    if max_year:
        try:
            query = query.filter(CarListing.year <= int(max_year))
        except ValueError:
            pass

    # Fuel type filter
    if fuel_type and fuel_type != "All":
        query = query.filter(CarListing.fuel_type == fuel_type)

    # Transmission filter
    if transmission and transmission != "All":
        query = query.filter(CarListing.transmission == transmission)

    # Owner filter
    if owner and owner != "All":
        query = query.filter(CarListing.owner == owner)

    # Sorting
    if sort == "price_asc":
        query = query.order_by(CarListing.asking_price.asc())
    elif sort == "price_desc":
        query = query.order_by(CarListing.asking_price.desc())
    elif sort == "year_desc":
        query = query.order_by(CarListing.year.desc())
    elif sort == "year_asc":
        query = query.order_by(CarListing.year.asc())
    elif sort == "km_asc":
        query = query.order_by(CarListing.km_driven.asc())
    else:  # newest by default
        query = query.order_by(CarListing.created_at.desc())

    listings = query.all()

    # Distinct brands for filter dropdown
    available_brands = [
        b[0] for b in db.session.query(CarListing.brand)
        .filter(CarListing.status == "active")
        .distinct()
        .order_by(CarListing.brand.asc())
        .all()
    ]

    filters = {
        "q": q,
        "brand": brand,
        "min_price": min_price,
        "max_price": max_price,
        "min_year": min_year,
        "max_year": max_year,
        "fuel_type": fuel_type,
        "transmission": transmission,
        "owner": owner,
        "sort": sort,
    }

    user_favourite_ids = set()
    if getattr(g, "user", None):
        user_favourite_ids = {f.listing_id for f in g.user.favourites}

    return render_template(
        "index.html",
        page="marketplace",
        listings=listings,
        filters=filters,
        total_count=len(listings),
        available_brands=available_brands,
        fuel_types=VALID_FUEL_TYPES,
        transmissions=VALID_TRANSMISSIONS,
        owners=VALID_OWNERS,
        user_favourite_ids=user_favourite_ids
    )


@app.route("/marketplace/<int:listing_id>")
def marketplace_detail(listing_id):
    """View detailed information for a single car listing."""
    listing = db.session.get(CarListing, listing_id)
    if not listing:
        flash("The requested car listing was not found.", "error")
        return redirect(url_for("marketplace"))

    # Inactive listings can only be viewed by the listing owner
    if listing.status != "active":
        if not getattr(g, "user", None) or g.user.id != listing.seller_id:
            flash("This car listing is no longer active in the marketplace.", "info")
            return redirect(url_for("marketplace"))

    is_favourited = False
    user_enquiry = None
    if getattr(g, "user", None):
        is_favourited = g.user.has_favourited(listing.id)
        user_enquiry = Inquiry.query.filter_by(
            listing_id=listing.id, buyer_id=g.user.id
        ).order_by(Inquiry.created_at.desc()).first()

    return render_template(
        "index.html",
        page="marketplace_detail",
        listing=listing,
        is_favourited=is_favourited,
        user_enquiry=user_enquiry
    )


@app.route("/marketplace/create", methods=["GET", "POST"])
def marketplace_create():
    """Create and publish a new vehicle listing in the database."""
    if not getattr(g, "user", None):
        flash("Please sign in to publish a car listing on the marketplace.", "info")
        return redirect(url_for("login", next=url_for("marketplace_create")))

    if request.method == "POST":
        token = request.form.get("csrf_token")
        if not token or token != session.get("_csrf_token"):
            flash("Security check failed. Please submit the form again.", "error")
            return render_template(
                "index.html",
                page="marketplace_form",
                action="create",
                form_data=request.form,
                fuel_types=VALID_FUEL_TYPES,
                transmissions=VALID_TRANSMISSIONS,
                owners=VALID_OWNERS
            )

        brand = request.form.get("brand", "").strip()
        model = request.form.get("model", "").strip()
        year_str = request.form.get("year", "").strip()
        km_str = request.form.get("km_driven", "").strip()
        fuel_type = request.form.get("fuel_type", "").strip()
        transmission = request.form.get("transmission", "").strip()
        owner = request.form.get("owner", "").strip()
        price_str = request.form.get("asking_price", "").strip()
        title = request.form.get("title", "").strip()
        description = request.form.get("description", "").strip()
        ai_price_str = request.form.get("ai_predicted_price", "").strip()

        errors = []
        if not brand:
            errors.append("Please specify the vehicle brand.")
        if not model:
            errors.append("Please specify the vehicle model.")

        year = None
        try:
            year = int(year_str)
            if year < 1980 or year > 2026:
                errors.append("Manufacturing year must be between 1980 and 2026.")
        except (ValueError, TypeError):
            errors.append("Please enter a valid numeric manufacturing year.")

        km_driven = None
        try:
            km_driven = int(km_str)
            if km_driven <= 0 or km_driven > 1000000:
                errors.append("Kilometers driven must be between 1 and 1,000,000 km.")
        except (ValueError, TypeError):
            errors.append("Please enter a valid number for kilometers driven.")

        if fuel_type not in VALID_FUEL_TYPES:
            errors.append("Please select a valid fuel type.")

        if transmission not in VALID_TRANSMISSIONS:
            errors.append("Please select a valid transmission.")

        if owner not in VALID_OWNERS:
            errors.append("Please select a valid ownership history.")

        asking_price = None
        try:
            asking_price = float(price_str)
            if asking_price < 10000 or asking_price > 500000000:
                errors.append("Asking price must be between ₹ 10,000 and ₹ 50,00,00,000.")
        except (ValueError, TypeError):
            errors.append("Please enter a valid numeric asking price in Rupees.")

        ai_predicted_price = None
        if ai_price_str:
            try:
                ai_predicted_price = float(ai_price_str)
            except (ValueError, TypeError):
                pass

        if not title:
            if year and brand and model:
                title = f"{year} {brand} {model}"
            else:
                errors.append("Please provide a listing title.")
        elif len(title) > 150:
            errors.append("Listing title must be 150 characters or less.")

        if errors:
            for error in errors:
                flash(error, "error")
            return render_template(
                "index.html",
                page="marketplace_form",
                action="create",
                form_data=request.form,
                fuel_types=VALID_FUEL_TYPES,
                transmissions=VALID_TRANSMISSIONS,
                owners=VALID_OWNERS
            )

        try:
            new_listing = CarListing(
                seller_id=g.user.id,
                brand=brand,
                model=model,
                year=year,
                km_driven=km_driven,
                fuel_type=fuel_type,
                transmission=transmission,
                owner=owner,
                asking_price=asking_price,
                ai_predicted_price=ai_predicted_price,
                title=title,
                description=description if description else None,
                status="active"
            )
            db.session.add(new_listing)
            db.session.commit()

            flash(f"🎉 Your listing for '{new_listing.title}' has been successfully published to the marketplace!", "success")
            return redirect(url_for("marketplace_detail", listing_id=new_listing.id))

        except Exception as e:
            db.session.rollback()
            print("Create Listing Error:", e)
            flash("An error occurred while publishing the listing. Please try again.", "error")
            return render_template(
                "index.html",
                page="marketplace_form",
                action="create",
                form_data=request.form,
                fuel_types=VALID_FUEL_TYPES,
                transmissions=VALID_TRANSMISSIONS,
                owners=VALID_OWNERS
            )

    # GET request - populate prefilled data if passed
    initial_data = {
        "brand": request.args.get("brand", "").strip(),
        "model": request.args.get("model", "").strip(),
        "year": request.args.get("year", "").strip(),
        "km_driven": request.args.get("kmDriven", request.args.get("km_driven", "")).strip(),
        "fuel_type": request.args.get("fuelType", request.args.get("fuel_type", "")).strip(),
        "transmission": request.args.get("transmission", "").strip(),
        "owner": request.args.get("owner", "").strip(),
        "asking_price": request.args.get("askingPrice", request.args.get("asking_price", "")).strip(),
        "ai_predicted_price": request.args.get("price", "").strip(),
        "title": request.args.get("title", "").strip(),
        "description": request.args.get("description", "").strip(),
    }

    return render_template(
        "index.html",
        page="marketplace_form",
        action="create",
        form_data=initial_data,
        fuel_types=VALID_FUEL_TYPES,
        transmissions=VALID_TRANSMISSIONS,
        owners=VALID_OWNERS
    )


@app.route("/marketplace/my-listings")
def my_listings():
    """Manage the authenticated user's own car listings."""
    if not getattr(g, "user", None):
        flash("Please sign in to view and manage your car listings.", "info")
        return redirect(url_for("login", next=url_for("my_listings")))

    listings = CarListing.query.filter_by(seller_id=g.user.id).order_by(CarListing.created_at.desc()).all()
    active_count = sum(1 for item in listings if item.status == "active")
    sold_count = sum(1 for item in listings if item.status == "sold")

    return render_template(
        "index.html",
        page="my_listings",
        listings=listings,
        active_count=active_count,
        sold_count=sold_count,
        total_count=len(listings)
    )


@app.route("/marketplace/<int:listing_id>/edit", methods=["GET", "POST"])
def marketplace_edit(listing_id):
    """Edit an existing listing (restricted strictly to listing owner)."""
    if not getattr(g, "user", None):
        flash("Please sign in to edit your listing.", "info")
        return redirect(url_for("login", next=url_for("marketplace_edit", listing_id=listing_id)))

    listing = db.session.get(CarListing, listing_id)
    if not listing:
        flash("The requested listing does not exist.", "error")
        return redirect(url_for("marketplace"))

    if listing.seller_id != g.user.id:
        flash("You do not have permission to edit this listing.", "error")
        return redirect(url_for("marketplace_detail", listing_id=listing.id))

    if request.method == "POST":
        token = request.form.get("csrf_token")
        if not token or token != session.get("_csrf_token"):
            flash("Security check failed. Please submit the form again.", "error")
            return render_template(
                "index.html",
                page="marketplace_form",
                action="edit",
                listing=listing,
                form_data=request.form,
                fuel_types=VALID_FUEL_TYPES,
                transmissions=VALID_TRANSMISSIONS,
                owners=VALID_OWNERS
            )

        brand = request.form.get("brand", "").strip()
        model = request.form.get("model", "").strip()
        year_str = request.form.get("year", "").strip()
        km_str = request.form.get("km_driven", "").strip()
        fuel_type = request.form.get("fuel_type", "").strip()
        transmission = request.form.get("transmission", "").strip()
        owner = request.form.get("owner", "").strip()
        price_str = request.form.get("asking_price", "").strip()
        title = request.form.get("title", "").strip()
        description = request.form.get("description", "").strip()
        status = request.form.get("status", "active").strip().lower()

        errors = []
        if not brand:
            errors.append("Please specify the vehicle brand.")
        if not model:
            errors.append("Please specify the vehicle model.")

        year = None
        try:
            year = int(year_str)
            if year < 1980 or year > 2026:
                errors.append("Manufacturing year must be between 1980 and 2026.")
        except (ValueError, TypeError):
            errors.append("Please enter a valid numeric manufacturing year.")

        km_driven = None
        try:
            km_driven = int(km_str)
            if km_driven <= 0 or km_driven > 1000000:
                errors.append("Kilometers driven must be between 1 and 1,000,000 km.")
        except (ValueError, TypeError):
            errors.append("Please enter a valid number for kilometers driven.")

        if fuel_type not in VALID_FUEL_TYPES:
            errors.append("Please select a valid fuel type.")

        if transmission not in VALID_TRANSMISSIONS:
            errors.append("Please select a valid transmission.")

        if owner not in VALID_OWNERS:
            errors.append("Please select a valid ownership history.")

        asking_price = None
        try:
            asking_price = float(price_str)
            if asking_price < 10000 or asking_price > 500000000:
                errors.append("Asking price must be between ₹ 10,000 and ₹ 50,00,00,000.")
        except (ValueError, TypeError):
            errors.append("Please enter a valid numeric asking price in Rupees.")

        if not title:
            if year and brand and model:
                title = f"{year} {brand} {model}"
            else:
                errors.append("Please provide a listing title.")
        elif len(title) > 150:
            errors.append("Listing title must be 150 characters or less.")

        if status not in ("active", "sold", "draft"):
            status = "active"

        if errors:
            for error in errors:
                flash(error, "error")
            return render_template(
                "index.html",
                page="marketplace_form",
                action="edit",
                listing=listing,
                form_data=request.form,
                fuel_types=VALID_FUEL_TYPES,
                transmissions=VALID_TRANSMISSIONS,
                owners=VALID_OWNERS
            )

        try:
            listing.brand = brand
            listing.model = model
            listing.year = year
            listing.km_driven = km_driven
            listing.fuel_type = fuel_type
            listing.transmission = transmission
            listing.owner = owner
            listing.asking_price = asking_price
            listing.title = title
            listing.description = description if description else None
            listing.status = status

            db.session.commit()
            flash(f"Listing '{listing.title}' updated successfully!", "success")
            return redirect(url_for("marketplace_detail", listing_id=listing.id))

        except Exception as e:
            db.session.rollback()
            print("Edit Listing Error:", e)
            flash("An error occurred while updating the listing.", "error")
            return render_template(
                "index.html",
                page="marketplace_form",
                action="edit",
                listing=listing,
                form_data=request.form,
                fuel_types=VALID_FUEL_TYPES,
                transmissions=VALID_TRANSMISSIONS,
                owners=VALID_OWNERS
            )

    # GET request - load listing data
    form_data = {
        "brand": listing.brand,
        "model": listing.model,
        "year": str(listing.year),
        "km_driven": str(listing.km_driven),
        "fuel_type": listing.fuel_type,
        "transmission": listing.transmission,
        "owner": listing.owner,
        "asking_price": str(int(listing.asking_price) if listing.asking_price.is_integer() else listing.asking_price),
        "ai_predicted_price": str(int(listing.ai_predicted_price) if listing.ai_predicted_price and listing.ai_predicted_price.is_integer() else (listing.ai_predicted_price or "")),
        "title": listing.title,
        "description": listing.description or "",
        "status": listing.status
    }

    return render_template(
        "index.html",
        page="marketplace_form",
        action="edit",
        listing=listing,
        form_data=form_data,
        fuel_types=VALID_FUEL_TYPES,
        transmissions=VALID_TRANSMISSIONS,
        owners=VALID_OWNERS
    )


@app.route("/marketplace/<int:listing_id>/status", methods=["POST"])
def marketplace_status(listing_id):
    """Toggle listing status between active and sold."""
    if not getattr(g, "user", None):
        flash("Please sign in to manage listing status.", "info")
        return redirect(url_for("login"))

    listing = db.session.get(CarListing, listing_id)
    if not listing:
        flash("Listing not found.", "error")
        return redirect(url_for("marketplace"))

    if listing.seller_id != g.user.id:
        flash("You do not have permission to modify this listing.", "error")
        return redirect(url_for("marketplace_detail", listing_id=listing.id))

    token = request.form.get("csrf_token")
    if not token or token != session.get("_csrf_token"):
        flash("Security check failed.", "error")
        return redirect(url_for("my_listings"))

    new_status = request.form.get("status", "").strip().lower()
    if new_status in ("active", "sold", "draft"):
        listing.status = new_status
        db.session.commit()
        status_label = "Marked as Sold" if new_status == "sold" else ("Marked as Active" if new_status == "active" else "Moved to Draft")
        flash(f"Listing '{listing.title}' was successfully {status_label.lower()}.", "success")

    next_target = request.form.get("next") or request.referrer or url_for("my_listings")
    return redirect(next_target)


@app.route("/marketplace/<int:listing_id>/delete", methods=["POST"])
def marketplace_delete(listing_id):
    """Delete a car listing permanently."""
    if not getattr(g, "user", None):
        flash("Please sign in to delete a listing.", "info")
        return redirect(url_for("login"))

    listing = db.session.get(CarListing, listing_id)
    if not listing:
        flash("Listing not found.", "error")
        return redirect(url_for("marketplace"))

    if listing.seller_id != g.user.id:
        flash("You do not have permission to delete this listing.", "error")
        return redirect(url_for("marketplace_detail", listing_id=listing.id))

    token = request.form.get("csrf_token")
    if not token or token != session.get("_csrf_token"):
        flash("Security check failed.", "error")
        return redirect(url_for("my_listings"))

    title = listing.title
    db.session.delete(listing)
    db.session.commit()
    flash(f"Listing '{title}' has been deleted.", "info")
    return redirect(url_for("my_listings"))


# ==========================================
# Favourites & Enquiries Routes
# ==========================================

@app.route("/marketplace/<int:listing_id>/favourite", methods=["POST"])
def marketplace_favourite_toggle(listing_id):
    """Add or remove a car listing from the user's saved favourites."""
    if not getattr(g, "user", None):
        flash("Please sign in to save vehicles to your favourites.", "info")
        return redirect(url_for("login", next=url_for("marketplace_detail", listing_id=listing_id)))

    token = request.form.get("csrf_token")
    if not token or token != session.get("_csrf_token"):
        flash("Security check failed. Please try again.", "error")
        return redirect(request.form.get("next") or request.referrer or url_for("marketplace"))

    listing = db.session.get(CarListing, listing_id)
    if not listing:
        flash("Car listing not found.", "error")
        return redirect(url_for("marketplace"))

    existing_fav = Favourite.query.filter_by(user_id=g.user.id, listing_id=listing.id).first()
    requested_action = request.form.get("action", "").strip().lower()

    if existing_fav:
        if requested_action == "add":
            flash(f"'{listing.title}' is already in your favourites.", "info")
        else:
            db.session.delete(existing_fav)
            db.session.commit()
            flash(f"Removed '{listing.title}' from your saved favourites.", "info")
    else:
        if requested_action == "remove":
            flash("This vehicle is not in your favourites.", "info")
        else:
            new_fav = Favourite(user_id=g.user.id, listing_id=listing.id)
            db.session.add(new_fav)
            try:
                db.session.commit()
                flash(f"❤️ Added '{listing.title}' to your saved favourites!", "success")
            except Exception:
                db.session.rollback()
                flash("Could not save to favourites. Please try again.", "error")

    next_url = request.form.get("next") or request.referrer or url_for("marketplace_favourites")
    return redirect(next_url)


@app.route("/marketplace/favourites/<int:favourite_id>/delete", methods=["POST"])
def marketplace_favourite_delete(favourite_id):
    """Remove a favourite entry directly by favourite ID (strictly restricted to owner)."""
    if not getattr(g, "user", None):
        flash("Please sign in to manage your favourites.", "info")
        return redirect(url_for("login"))

    token = request.form.get("csrf_token")
    if not token or token != session.get("_csrf_token"):
        flash("Security check failed.", "error")
        return redirect(url_for("marketplace_favourites"))

    fav = db.session.get(Favourite, favourite_id)
    if not fav:
        flash("Saved vehicle record not found.", "error")
        return redirect(url_for("marketplace_favourites"))

    if fav.user_id != g.user.id:
        flash("You do not have permission to remove this favourite.", "error")
        return redirect(url_for("marketplace_favourites"))

    title = fav.listing.title if fav.listing else "Vehicle"
    db.session.delete(fav)
    db.session.commit()
    flash(f"Removed '{title}' from your favourites.", "info")
    return redirect(url_for("marketplace_favourites"))


@app.route("/marketplace/favourites")
def marketplace_favourites():
    """Display all saved / bookmarked vehicles for the authenticated user."""
    if not getattr(g, "user", None):
        flash("Please sign in to view your saved favourites.", "info")
        return redirect(url_for("login", next=url_for("marketplace_favourites")))

    favs = Favourite.query.filter_by(user_id=g.user.id).order_by(Favourite.created_at.desc()).all()
    active_count = sum(1 for f in favs if f.listing and f.listing.is_active)

    return render_template(
        "index.html",
        page="favourites",
        favourites=favs,
        active_count=active_count,
        total_count=len(favs)
    )


@app.route("/marketplace/<int:listing_id>/enquire", methods=["POST"])
def marketplace_enquire(listing_id):
    """Send an enquiry or price offer about an active car listing to its seller."""
    if not getattr(g, "user", None):
        flash("Please sign in to send an enquiry to the seller.", "info")
        return redirect(url_for("login", next=url_for("marketplace_detail", listing_id=listing_id)))

    token = request.form.get("csrf_token")
    if not token or token != session.get("_csrf_token"):
        flash("Security check failed. Please submit your enquiry again.", "error")
        return redirect(url_for("marketplace_detail", listing_id=listing_id))

    listing = db.session.get(CarListing, listing_id)
    if not listing:
        flash("The requested vehicle listing was not found.", "error")
        return redirect(url_for("marketplace"))

    # Prevent sellers from enquiring on their own listing
    if listing.seller_id == g.user.id:
        flash("You cannot send an enquiry on your own vehicle listing.", "error")
        return redirect(url_for("marketplace_detail", listing_id=listing.id))

    # Check listing status is active
    if listing.status != "active":
        flash("This vehicle listing is no longer active and cannot accept new enquiries.", "error")
        return redirect(url_for("marketplace_detail", listing_id=listing.id))

    message = request.form.get("message", "").strip()
    if not message:
        flash("Please write a message for your enquiry.", "error")
        return redirect(url_for("marketplace_detail", listing_id=listing.id))

    if len(message) < 5:
        flash("Enquiry message must be at least 5 characters long.", "error")
        return redirect(url_for("marketplace_detail", listing_id=listing.id))

    if len(message) > 3000:
        flash("Enquiry message must not exceed 3,000 characters.", "error")
        return redirect(url_for("marketplace_detail", listing_id=listing.id))

    offer_price = None
    offer_str = request.form.get("offer_price", "").strip()
    if offer_str:
        try:
            offer_price = float(offer_str)
            if offer_price < 1000 or offer_price > 500000000:
                flash("Offer price must be between ₹ 1,000 and ₹ 50,00,00,000.", "error")
                return redirect(url_for("marketplace_detail", listing_id=listing.id))
        except (ValueError, TypeError):
            flash("Please enter a valid numeric offer price.", "error")
            return redirect(url_for("marketplace_detail", listing_id=listing.id))

    new_inquiry = Inquiry(
        listing_id=listing.id,
        buyer_id=g.user.id,
        seller_id=listing.seller_id,
        message=message,
        offer_price=offer_price,
        status="unread"
    )
    db.session.add(new_inquiry)
    db.session.flush()

    initial_msg = InquiryMessage(
        inquiry_id=new_inquiry.id,
        sender_id=g.user.id,
        message=message,
        is_read=False,
        created_at=new_inquiry.created_at
    )
    db.session.add(initial_msg)
    db.session.commit()

    flash(f"🎉 Your enquiry for '{listing.title}' has been successfully sent to {listing.seller.full_name}!", "success")
    return redirect(url_for("marketplace_detail", listing_id=listing.id))


@app.route("/marketplace/enquiries")
def marketplace_enquiries():
    """Display enquiries received as a seller and sent as a prospective buyer."""
    if not getattr(g, "user", None):
        flash("Please sign in to view your enquiries.", "info")
        return redirect(url_for("login", next=url_for("marketplace_enquiries")))

    received_inquiries = Inquiry.query.filter_by(seller_id=g.user.id).order_by(Inquiry.created_at.desc()).all()
    sent_inquiries = Inquiry.query.filter_by(buyer_id=g.user.id).order_by(Inquiry.created_at.desc()).all()

    unread_count = sum(1 for i in received_inquiries if i.status == "unread")

    return render_template(
        "index.html",
        page="enquiries",
        received_inquiries=received_inquiries,
        sent_inquiries=sent_inquiries,
        unread_count=unread_count,
        total_received=len(received_inquiries),
        total_sent=len(sent_inquiries)
    )


@app.route("/marketplace/enquiries/<int:inquiry_id>/status", methods=["POST"])
def marketplace_inquiry_status(inquiry_id):
    """Update status of an enquiry (read, replied, closed). Only recipient/seller or sender can update."""
    if not getattr(g, "user", None):
        flash("Please sign in to manage enquiry status.", "info")
        return redirect(url_for("login"))

    token = request.form.get("csrf_token")
    if not token or token != session.get("_csrf_token"):
        flash("Security check failed.", "error")
        return redirect(url_for("marketplace_enquiries"))

    inquiry = db.session.get(Inquiry, inquiry_id)
    if not inquiry:
        flash("Enquiry not found.", "error")
        return redirect(url_for("marketplace_enquiries"))

    # Strictly check access: only the seller (recipient) or buyer (sender) can modify status
    if inquiry.seller_id != g.user.id and inquiry.buyer_id != g.user.id:
        flash("You do not have permission to manage this enquiry.", "error")
        return redirect(url_for("marketplace_enquiries"))

    new_status = request.form.get("status", "").strip().lower()
    valid_statuses = ["unread", "read", "replied", "closed"]
    if new_status in valid_statuses:
        inquiry.status = new_status
        db.session.commit()
        flash(f"Enquiry #{inquiry.id} status updated to '{new_status.capitalize()}'.", "success")
    else:
        flash("Invalid status selected.", "error")

    next_url = request.form.get("next") or request.referrer or url_for("marketplace_enquiries")
    return redirect(next_url)


@app.route("/marketplace/enquiries/<int:inquiry_id>")
def marketplace_enquiry_thread(inquiry_id):
    """View private in-app conversation thread between buyer and seller for an enquiry."""
    if not getattr(g, "user", None):
        flash("Please sign in to view this conversation.", "info")
        return redirect(url_for("login", next=url_for("marketplace_enquiry_thread", inquiry_id=inquiry_id)))

    inquiry = db.session.get(Inquiry, inquiry_id)
    if not inquiry:
        flash("The requested conversation or enquiry was not found.", "error")
        return redirect(url_for("marketplace_enquiries"))

    # Strict access control: only the buyer or seller can view this conversation
    if inquiry.buyer_id != g.user.id and inquiry.seller_id != g.user.id:
        flash("You do not have permission to view this conversation.", "error")
        return redirect(url_for("marketplace_enquiries"))

    # If legacy enquiry has no message rows, seed the initial message row
    if not inquiry.messages and inquiry.message:
        initial_msg = InquiryMessage(
            inquiry_id=inquiry.id,
            sender_id=inquiry.buyer_id,
            message=inquiry.message,
            is_read=(inquiry.status in ["read", "replied", "closed"]),
            created_at=inquiry.created_at
        )
        db.session.add(initial_msg)
        db.session.commit()

    # Automatically mark incoming messages from counterpart as read
    marked_any = False
    for msg in inquiry.messages:
        if msg.sender_id != g.user.id and not msg.is_read:
            msg.is_read = True
            marked_any = True

    # If seller is viewing an unread inquiry, update status to read
    if g.user.id == inquiry.seller_id and inquiry.status == "unread":
        inquiry.status = "read"
        marked_any = True

    if marked_any:
        db.session.commit()

    is_buyer = (g.user.id == inquiry.buyer_id)
    counterpart = inquiry.seller if is_buyer else inquiry.buyer
    counterpart_role = "Seller" if is_buyer else "Buyer"

    return render_template(
        "index.html",
        page="enquiry_thread",
        inquiry=inquiry,
        listing=inquiry.listing,
        messages=inquiry.conversation_messages,
        is_buyer=is_buyer,
        counterpart=counterpart,
        counterpart_role=counterpart_role
    )


@app.route("/marketplace/enquiries/<int:inquiry_id>/reply", methods=["POST"])
def marketplace_enquiry_reply(inquiry_id):
    """Post a reply message in an active buyer-seller enquiry thread."""
    if not getattr(g, "user", None):
        flash("Please sign in to send a message.", "info")
        return redirect(url_for("login", next=url_for("marketplace_enquiry_thread", inquiry_id=inquiry_id)))

    token = request.form.get("csrf_token")
    if not token or token != session.get("_csrf_token"):
        flash("Security check failed. Please submit your message again.", "error")
        return redirect(url_for("marketplace_enquiry_thread", inquiry_id=inquiry_id))

    inquiry = db.session.get(Inquiry, inquiry_id)
    if not inquiry:
        flash("Enquiry not found.", "error")
        return redirect(url_for("marketplace_enquiries"))

    # Strict access control: only the buyer or seller can post
    if inquiry.buyer_id != g.user.id and inquiry.seller_id != g.user.id:
        flash("You do not have permission to reply to this conversation.", "error")
        return redirect(url_for("marketplace_enquiries"))

    message_text = request.form.get("message", "").strip()
    if not message_text:
        flash("Message cannot be empty.", "error")
        return redirect(url_for("marketplace_enquiry_thread", inquiry_id=inquiry_id))

    if len(message_text) < 1:
        flash("Message cannot be empty.", "error")
        return redirect(url_for("marketplace_enquiry_thread", inquiry_id=inquiry_id))

    if len(message_text) > 3000:
        flash("Message must not exceed 3,000 characters.", "error")
        return redirect(url_for("marketplace_enquiry_thread", inquiry_id=inquiry_id))

    # If legacy enquiry has no message rows, seed initial message first
    if not inquiry.messages and inquiry.message:
        initial_msg = InquiryMessage(
            inquiry_id=inquiry.id,
            sender_id=inquiry.buyer_id,
            message=inquiry.message,
            is_read=(inquiry.status in ["read", "replied", "closed"]),
            created_at=inquiry.created_at
        )
        db.session.add(initial_msg)
        db.session.flush()

    new_msg = InquiryMessage(
        inquiry_id=inquiry.id,
        sender_id=g.user.id,
        message=message_text,
        is_read=False,
        created_at=utcnow()
    )
    db.session.add(new_msg)

    # Status updates
    if g.user.id == inquiry.seller_id:
        inquiry.status = "replied"
    elif g.user.id == inquiry.buyer_id and inquiry.status in ("replied", "read"):
        inquiry.status = "unread"

    db.session.commit()
    flash("Message sent successfully!", "success")
    return redirect(url_for("marketplace_enquiry_thread", inquiry_id=inquiry_id))



# ==========================================
# Run Flask Application
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True
    )
