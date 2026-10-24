from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import math

app = Flask(__name__)


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
# Load and cache Dataset & Brand/Model mapping
# ==========================================

def load_dataset_and_mapping():
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
        return df, mapping
    except Exception as e:
        print("Error loading dataset and brand models mapping:", e)
        return pd.DataFrame(), {}

CARS_DATASET, BRAND_MODELS = load_dataset_and_mapping()
ALL_BRANDS = sorted(list(BRAND_MODELS.keys()))


# ==========================================
# SIMILAR CARS & COMPARABLES ALGORITHM
# ==========================================

def compute_similarity(target, candidate):
    """
    Normalized, bounded similarity score (0.0% to 100.0%).
    Weights:
      With asking price: Model (20%), Year (18%), Mileage Base (22%), Transmission (12%), Fuel (12%), Owner (6%), Price (10%).
      Without asking price: Model (22%), Year (20%), Mileage Base (24%), Transmission (14%), Fuel (14%), Owner (6%).
    Excess mileage penalty:
      If |km_target - km_cand| > 35,000 km, applies a calibrated deduction (up to 20-25%)
      so large mileage disparities (e.g. 97,000 km gap) cannot be masked by matching categorical fields.
    """
    has_target_price = target.get("asking_price") is not None and target["asking_price"] > 0

    if has_target_price:
        weights = {
            "model": 0.20,
            "year": 0.18,
            "km": 0.22,
            "transmission": 0.12,
            "fuel": 0.12,
            "owner": 0.06,
            "price": 0.10
        }
        max_penalty = 20.0
    else:
        weights = {
            "model": 0.22,
            "year": 0.20,
            "km": 0.24,
            "transmission": 0.14,
            "fuel": 0.14,
            "owner": 0.06,
            "price": 0.00
        }
        max_penalty = 25.0

    # 1. Model match component
    if str(candidate["Brand"]).strip().lower() == str(target["brand"]).strip().lower() and \
       str(candidate["model"]).strip().lower() == str(target["model"]).strip().lower():
        s_model = 1.0
    elif str(candidate["Brand"]).strip().lower() == str(target["brand"]).strip().lower():
        s_model = 0.40  # Same brand broader fallback
    else:
        s_model = 0.00  # Market segment fallback

    # 2. Year component (10 years difference = 0 score)
    year_diff = abs(int(candidate["Year"]) - int(target["year"]))
    s_year = max(0.0, 1.0 - min(1.0, year_diff / 10.0))

    # 3. KM component (100,000 km difference = 0 score)
    km_diff = abs(float(candidate["kmDriven"]) - float(target["km"]))
    s_km = max(0.0, 1.0 - min(1.0, km_diff / 100000.0))

    # 4. Transmission component
    trans_c = str(candidate["Transmission"]).strip().lower()
    trans_t = str(target["transmission"]).strip().lower()
    s_trans = 1.0 if trans_c == trans_t else 0.0

    # 5. Fuel Type component
    fuel_c = str(candidate["FuelType"]).strip().lower()
    fuel_t = str(target["fuel"]).strip().lower()
    s_fuel = 1.0 if fuel_c == fuel_t else 0.0

    # 6. Owner component
    owner_c = str(candidate["Owner"]).strip().lower()
    owner_t = str(target["owner"]).strip().lower()
    is_owner_match = (
        ("first" in owner_c and "first" in owner_t) or
        ("second" in owner_c and "second" in owner_t) or
        ("third" in owner_c and "third" in owner_t)
    )
    s_owner = 1.0 if is_owner_match else 0.0

    # 7. Price component (only applied if valid positive asking price provided)
    if has_target_price:
        price_diff_ratio = abs(float(candidate["AskPrice"]) - float(target["asking_price"])) / float(target["asking_price"])
        s_price = max(0.0, 1.0 - min(1.0, price_diff_ratio))
    else:
        s_price = 0.0

    base_score = (
        weights["model"] * s_model +
        weights["year"] * s_year +
        weights["km"] * s_km +
        weights["transmission"] * s_trans +
        weights["fuel"] * s_fuel +
        weights["owner"] * s_owner +
        weights["price"] * s_price
    ) * 100.0

    # Excess mileage penalty if gap exceeds 35,000 km
    if km_diff > 35000:
        excess = km_diff - 35000
        penalty = min(max_penalty, (excess / 65000.0) * max_penalty)
    else:
        penalty = 0.0

    final_score = max(0.0, min(100.0, round(base_score - penalty, 1)))
    km_closeness_pct = round(s_km * 100.0, 1)

    # Human-readable mileage comparison text
    kd = int(candidate["kmDriven"] - target["km"])
    if abs(kd) < 2500:
        km_comp_text = f"Similar mileage (~{abs(kd):,} km diff)"
    elif kd < 0:
        km_comp_text = f"{abs(kd):,} km less driven than target"
    else:
        km_comp_text = f"{kd:,} km more driven than target"

    return final_score, km_closeness_pct, km_diff, km_comp_text

def find_similar_cars(target, top_n=5):
    """
    Searches CARS_DATASET for top_n comparable vehicles using transparent multi-tier fallback.
    """
    if CARS_DATASET.empty:
        return {"matches": [], "tier": "empty", "exact_count": 0, "stats": {"median_price": 0, "min_price": 0, "max_price": 0, "count": 0}}

    # Tier 1: Exact Brand & Model
    exact_mask = (CARS_DATASET["Brand"].str.lower() == target["brand"].lower()) & (CARS_DATASET["model"].str.lower() == target["model"].lower())
    exact_df = CARS_DATASET[exact_mask].copy()

    tier = "exact"
    exact_count = len(exact_df)

    if exact_count >= top_n:
        candidates = exact_df
    else:
        # Tier 2: Same Brand fallback
        brand_mask = (CARS_DATASET["Brand"].str.lower() == target["brand"].lower())
        brand_df = CARS_DATASET[brand_mask].copy()
        if len(brand_df) >= top_n:
            candidates = brand_df
            tier = "partial_brand_supplement" if exact_count > 0 else "brand_only"
        else:
            candidates = CARS_DATASET.copy()
            tier = "broader_segment"

    scores = []
    km_closeness_list = []
    km_diff_list = []
    km_text_list = []
    differences_list = []

    for idx, row in candidates.iterrows():
        sc, km_close, km_diff_val, km_text = compute_similarity(target, row)
        scores.append(sc)
        km_closeness_list.append(km_close)
        km_diff_list.append(km_diff_val)
        km_text_list.append(km_text)

        diffs = []
        is_exact_model = (row["model"].lower() == target["model"].lower())
        if not is_exact_model:
            diffs.append(f"Model: {row['model']}")

        yd = row["Year"] - target["year"]
        if yd == 0:
            diffs.append("Same Year")
        elif yd > 0:
            diffs.append(f"+{yd} yr newer")
        else:
            diffs.append(f"{abs(yd)} yr older")

        diffs.append(km_text)

        if row["Transmission"].lower() != target["transmission"].lower():
            diffs.append(f"{row['Transmission']}")
        else:
            diffs.append(f"Matching {row['Transmission']}")

        if row["FuelType"].lower() != target["fuel"].lower():
            diffs.append(f"{row['FuelType']}")

        differences_list.append(" • ".join(diffs[:3]))

    candidates["similarity_score"] = scores
    candidates["km_closeness"] = km_closeness_list
    candidates["km_diff_abs"] = km_diff_list
    candidates["km_comparison_text"] = km_text_list
    candidates["differences_summary"] = differences_list
    candidates["is_exact_model"] = (candidates["model"].str.lower() == target["model"].lower())

    sorted_candidates = candidates.sort_values(by=["is_exact_model", "similarity_score"], ascending=[False, False])
    top_matches = sorted_candidates.head(top_n).copy()

    # Predict ML Price & Quality for top matches
    model_inputs = pd.DataFrame({
        "Brand": top_matches["Brand"],
        "model": top_matches["model"],
        "Transmission": top_matches["Transmission"],
        "Owner": top_matches["Owner"],
        "FuelType": top_matches["FuelType"],
        "Year": top_matches["Year"],
        "Age": top_matches["Age"],
        "kmDriven": top_matches["kmDriven"]
    })

    top_matches["ml_price"] = [round(max(0, float(p))) for p in regression_model.predict(model_inputs)]
    top_matches["ml_quality"] = classification_model.predict(model_inputs)

    # Historical price statistics strictly from relevant displayed comparables
    hist_prices = top_matches["AskPrice"].tolist()
    if hist_prices:
        import numpy as np
        hist_stats = {
            "median_price": round(float(np.median(hist_prices))),
            "min_price": int(min(hist_prices)),
            "max_price": int(max(hist_prices)),
            "count": len(hist_prices)
        }
    else:
        hist_stats = {"median_price": 0, "min_price": 0, "max_price": 0, "count": 0}

    return {
        "matches": top_matches.to_dict(orient="records"),
        "tier": tier,
        "exact_count": exact_count,
        "stats": hist_stats
    }


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
        page="buyer"
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
    brand_a = request.args.get("brandA", "").strip()
    model_a = request.args.get("modelA", "").strip()
    year_a = request.args.get("yearA", "").strip()
    km_a = request.args.get("kmA", "").strip()
    trans_a = request.args.get("transmissionA", "").strip()
    owner_a = request.args.get("ownerA", "").strip()
    fuel_a = request.args.get("fuelA", "").strip()

    brand_b = request.args.get("brandB", "").strip()
    model_b = request.args.get("modelB", "").strip()
    year_b = request.args.get("yearB", "").strip()
    km_b = request.args.get("kmB", "").strip()
    trans_b = request.args.get("transmissionB", "").strip()
    owner_b = request.args.get("ownerB", "").strip()
    fuel_b = request.args.get("fuelB", "").strip()

    # Safely validate brand & model if provided
    if brand_a and brand_a not in BRAND_MODELS:
        brand_a, model_a = "", ""
    elif model_a and model_a not in BRAND_MODELS.get(brand_a, []):
        model_a = ""

    if brand_b and brand_b not in BRAND_MODELS:
        brand_b, model_b = "", ""
    elif model_b and model_b not in BRAND_MODELS.get(brand_b, []):
        model_b = ""

    return render_template(
        "index.html",
        page="compare",
        brandA=brand_a,
        modelA=model_a,
        yearA=year_a,
        kmA=km_a,
        transmissionA=trans_a,
        ownerA=owner_a,
        fuelA=fuel_a,
        brandB=brand_b,
        modelB=model_b,
        yearB=year_b,
        kmB=km_b,
        transmissionB=trans_b,
        ownerB=owner_b,
        fuelB=fuel_b
    )


# ==========================================
# Similar Cars & Market Comparables
# ==========================================

@app.route("/comparables")
def comparables():
    brand = request.args.get("brand", "").strip()
    model = request.args.get("model", "").strip()
    year_text = request.args.get("year", "").strip()
    km_text = request.args.get("kmDriven", "").strip()
    transmission = request.args.get("transmission", "").strip()
    owner = request.args.get("owner", "").strip()
    fuel_type = request.args.get("fuelType", "").strip()
    asking_price_text = request.args.get("askingPrice", "").strip()

    has_search = bool(brand or model or year_text or km_text)

    form_data = {
        "brand": brand,
        "model": model,
        "year": year_text,
        "kmDriven": km_text,
        "transmission": transmission,
        "owner": owner,
        "fuelType": fuel_type,
        "askingPrice": asking_price_text
    }

    if not has_search:
        return render_template(
            "index.html",
            page="comparables",
            has_results=False,
            **form_data
        )

    # Validate target inputs
    error = None
    if not brand or not model:
        error = "Please select both brand and model."
    elif brand not in BRAND_MODELS or model not in BRAND_MODELS.get(brand, []):
        error = "Please select a valid brand and model combination."
    elif not year_text:
        error = "Please enter the manufacturing year."
    elif not km_text:
        error = "Please enter kilometers driven."
    elif not transmission:
        error = "Please select a transmission type."
    elif not owner:
        error = "Please select an owner type."
    elif not fuel_type:
        error = "Please select a fuel type."

    if error:
        return render_template(
            "index.html",
            page="comparables",
            has_results=False,
            error=error,
            **form_data
        )

    try:
        year = int(year_text)
        km = float(km_text)
        if year < 1980 or year > 2024:
            return render_template("index.html", page="comparables", has_results=False, error="Manufacturing year must be between 1980 and 2024.", **form_data)
        if not math.isfinite(km) or km <= 0:
            return render_template("index.html", page="comparables", has_results=False, error="Kilometers driven must be greater than 0.", **form_data)

        asking_price = None
        if asking_price_text:
            try:
                ap = float(asking_price_text)
                if math.isfinite(ap) and ap > 0:
                    asking_price = ap
            except ValueError:
                pass

        # Target car valuation & quality
        age = 2024 - year
        target_df = pd.DataFrame({
            "Brand": [brand],
            "model": [model],
            "Transmission": [transmission],
            "Owner": [owner],
            "FuelType": [fuel_type],
            "Year": [year],
            "Age": [age],
            "kmDriven": [km]
        })
        target_pred_price = round(max(0, float(regression_model.predict(target_df)[0])))
        target_quality = str(classification_model.predict(target_df)[0])

        target_specs = {
            "brand": brand,
            "model": model,
            "year": year,
            "km": km,
            "transmission": transmission,
            "owner": owner,
            "fuel": fuel_type,
            "asking_price": asking_price
        }

        comparables_data = find_similar_cars(target_specs, top_n=5)

        # Market position text if asking price is supplied
        target_market_position = None
        if asking_price and comparables_data["stats"]["median_price"]:
            median = comparables_data["stats"]["median_price"]
            diff = asking_price - median
            pct = abs(diff / median) * 100
            if abs(diff) < 15000:
                target_market_position = "Target asking price aligns closely with the historical dataset median for comparable cars."
            elif diff < 0:
                target_market_position = f"Target asking price is ₹{abs(round(diff)):,} ({pct:.1f}%) below the historical dataset median."
            else:
                target_market_position = f"Target asking price is ₹{round(diff):,} ({pct:.1f}%) above the historical dataset median."

        return render_template(
            "index.html",
            page="comparables",
            has_results=True,
            target_valuation=target_pred_price,
            target_quality=target_quality,
            target_market_position=target_market_position,
            comparables=comparables_data["matches"],
            tier=comparables_data["tier"],
            exact_count=comparables_data["exact_count"],
            stats=comparables_data["stats"],
            **form_data
        )

    except Exception as e:
        print("Comparables processing error:", e)
        return render_template(
            "index.html",
            page="comparables",
            has_results=False,
            error="An error occurred while finding similar cars. Please check your inputs.",
            **form_data
        )


# ==========================================
# Seller Mode
# ==========================================

@app.route("/seller")
def seller():

    return render_template(
        "index.html",
        page="seller"
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
                page="seller",
                error="Please select both brand and model.",
                **form_data
            )

        if brand not in BRAND_MODELS or model not in BRAND_MODELS.get(brand, []):

            return render_template(
                "index.html",
                page="seller",
                error="Please select a valid brand and model combination.",
                **form_data
            )


        if not year_text:

            return render_template(
                "index.html",
                page="seller",
                error="Please enter the manufacturing year.",
                **form_data
            )


        if not km_text:

            return render_template(
                "index.html",
                page="seller",
                error="Please enter kilometers driven.",
                **form_data
            )

        if not asking_price_text:

            return render_template(
                "index.html",
                page="seller",
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
                page="seller",
                error="Please enter a valid asking price.",
                **form_data
            )

        if not math.isfinite(asking_price) or asking_price <= 0:

            return render_template(
                "index.html",
                page="seller",
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
                page="seller",
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
                page="seller",
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
                page="seller",
                error="Please select a transmission type.",
                **form_data
            )


        if not owner:

            return render_template(
                "index.html",
                page="seller",
                error="Please select an owner type.",
                **form_data
            )


        if not fuel_type:

            return render_template(
                "index.html",
                page="seller",
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

            page="seller",

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

            page="seller",

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

            page="seller",

            error=(
                "Something went wrong while processing "
                "the prediction. Please check your inputs."
            ),

            **form_data

        )


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