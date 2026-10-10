import streamlit as st
import pandas as pd
import pickle

# =====================================================
# Page Configuration
# =====================================================
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# Load Model
# =====================================================
@st.cache_resource
def load_artifacts():
    with open("decision_tree_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("features.pkl", "rb") as f:
        features = pickle.load(f)

    return model, features


model, features = load_artifacts()

# =====================================================
# Premium Automotive UI Styling
# =====================================================
st.markdown(
    """
    <style>
    .stApp {
        background: #0b1120;
        color: #e5e7eb;
    }
    #MainMenu, footer, header { visibility: hidden; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101827, #172033);
        border-right: 1px solid #273449;
    }
    [data-testid="stSidebar"] * { color: #e5e7eb !important; }

    .hero {
        background:
            linear-gradient(90deg, rgba(7, 12, 24, .96) 0%,
            rgba(7, 12, 24, .82) 48%, rgba(7, 12, 24, .20) 100%),
            url("https://images.unsplash.com/photo-1503376780353-7e6692767b70?auto=format&fit=crop&w=1800&q=85");
        background-position: center 58%;
        background-size: cover;
        border: 1px solid #344256;
        border-radius: 22px;
        padding: 42px 34px;
        min-height: 255px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 18px 45px rgba(0,0,0,.25);
    }
    .hero h1 {
        font-size: clamp(28px, 3vw, 42px);
        font-weight: 850;
        margin: 0;
        color: #ffffff;
    }
    .hero p { color: #d1d9e6; margin-top: 12px; max-width: 600px; }
    .eyebrow {
        color: #f59e0b;
        font-size: 12px;
        letter-spacing: 3px;
        font-weight: 800;
        margin-bottom: 12px;
    }
    .section-card {
        background: #121c2d;
        border: 1px solid #29364a;
        border-radius: 16px;
        padding: 21px;
        margin-bottom: 16px;
        box-shadow: 0 8px 24px rgba(0,0,0,.12);
    }
    .section-title { color: #f8fafc; font-size: 20px; font-weight: 850; }
    .section-subtitle { color: #a7b4c7; font-size: 13px; margin-top: 5px; }
    .price-card {
        background: linear-gradient(135deg, #12352f, #102b28);
        border: 1px solid #267768;
        border-radius: 18px;
        padding: 28px 16px;
        text-align: center;
        margin: 12px 0;
    }
    .price-label { color: #6ee7b7; font-size: 13px; font-weight: 800; letter-spacing: 1px; }
    .price-value { color: #a7f3d0; font-size: 35px; font-weight: 900; margin-top: 8px; }
    .placeholder {
        background: #121c2d;
        border: 1px dashed #475569;
        border-radius: 18px;
        padding: 40px 20px;
        text-align: center;
    }
    .placeholder h3 { color: #f8fafc; }
    .placeholder p { color: #a7b4c7; }
    div.stButton > button, div.stFormSubmitButton > button {
        min-height: 46px;
        border-radius: 10px;
        font-weight: 800;
        border: 1px solid #f59e0b;
    }
    div.stFormSubmitButton > button {
        background: #f59e0b;
        color: #111827;
    }
    [data-testid="stNumberInput"] label,
    [data-testid="stSelectbox"] label { color: #dbe4f0 !important; }
    [data-testid="stMetric"] {
        background: #121c2d;
        padding: 14px;
        border-radius: 12px;
        border: 1px solid #29364a;
    }
    .car-caption {
        color: #a7b4c7;
        font-size: 12px;
        text-align: center;
        margin-top: -8px;
        margin-bottom: 18px;
    }
    @media (max-width: 700px) {
        .hero { padding: 28px 22px; min-height: 220px; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# Hero Header
# =====================================================
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">SMART MOBILITY • MACHINE LEARNING</div>
        <h1>Find the value of your next car.</h1>
        <p>
            Explore vehicle specifications and estimate automobile value
            with your Decision Tree machine-learning model.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# Featured vehicle gallery
st.markdown("### Explore the garage")
car1, car2, car3 = st.columns(3, gap="medium")
with car1:
    st.image(
        "https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?auto=format&fit=crop&w=900&q=80",
        use_container_width=True
    )
    st.markdown('<div class="car-caption">SPORTS & PERFORMANCE</div>', unsafe_allow_html=True)
with car2:
    st.image(
        "https://images.unsplash.com/photo-1503376780353-7e6692767b70?auto=format&fit=crop&w=900&q=80",
        use_container_width=True
    )
    st.markdown('<div class="car-caption">PREMIUM AUTOMOBILES</div>', unsafe_allow_html=True)
with car3:
    st.image(
        "https://images.unsplash.com/photo-1552519507-da3b142c6e3d?auto=format&fit=crop&w=900&q=80",
        use_container_width=True
    )
    st.markdown('<div class="car-caption">MODERN DESIGN</div>', unsafe_allow_html=True)


# =====================================================
# Sidebar
# =====================================================
with st.sidebar:
    st.markdown("## 🚘 AUTO VALUE AI")
    st.caption("Your vehicle valuation studio")
    st.divider()

    st.markdown("### MODEL ENGINE")
    st.write("Decision Tree Regressor")

    st.markdown("### VEHICLE DATA")
    st.write("Vehicle specifications and engine characteristics")

    st.divider()
    st.info(
        "Complete the vehicle details, then run your price estimate."
    )

# =====================================================
# Input Form
# =====================================================
left, right = st.columns([1.25, 1], gap="large")

with left:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">🚘 Vehicle Information</div>
            <div class="section-subtitle">
                Enter the specifications of the automobile.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("car_prediction_form"):
        c1, c2 = st.columns(2)

        with c1:
            symboling = st.number_input(
                "Insurance Risk Rating",
                min_value=-3, max_value=3, value=0
            )

            normalized_losses = st.number_input(
                "Normalized Losses",
                min_value=0, value=120
            )

            make = st.selectbox(
                "Manufacturer",
                [
                    "alfa-romero", "audi", "bmw", "chevrolet",
                    "dodge", "honda", "isuzu", "jaguar",
                    "mazda", "mercedes-benz", "mitsubishi",
                    "nissan", "peugot", "plymouth", "porsche",
                    "renault", "saab", "subaru", "toyota",
                    "volkswagen", "volvo"
                ]
            )

            fuel_type = st.selectbox(
                "Fuel Type", ["gas", "diesel"]
            )

            aspiration = st.selectbox(
                "Aspiration", ["std", "turbo"]
            )

            num_doors = st.selectbox(
                "Number of Doors", ["four", "two"]
            )

            body_style = st.selectbox(
                "Body Style",
                [
                    "sedan", "hatchback", "wagon",
                    "hardtop", "convertible"
                ]
            )

            drive_wheels = st.selectbox(
                "Drive Wheels", ["fwd", "rwd", "4wd"]
            )

            engine_location = st.selectbox(
                "Engine Location", ["front", "rear"]
            )

            wheel_base = st.number_input(
                "Wheel Base", min_value=80.0, value=95.0
            )

            length = st.number_input(
                "Vehicle Length", min_value=100.0, value=170.0
            )

        with c2:
            width = st.number_input(
                "Vehicle Width", min_value=50.0, value=65.0
            )

            height = st.number_input(
                "Vehicle Height", min_value=40.0, value=53.0
            )

            curb_weight = st.number_input(
                "Curb Weight", min_value=100, value=2500
            )

            engine_type = st.selectbox(
                "Engine Type",
                ["dohc", "ohcv", "ohc", "l", "rotor",
                 "ohcf", "dohcv"]
            )

            num_cylinders = st.selectbox(
                "Number of Cylinders",
                ["two", "three", "four", "five",
                 "six", "eight", "twelve"]
            )

            engine_size = st.number_input(
                "Engine Size", min_value=50, value=130
            )

            fuel_system = st.selectbox(
                "Fuel System",
                ["mpfi", "2bbl", "1bbl", "spdi",
                 "4bbl", "idi", "mfi", "spfi"]
            )

            bore = st.number_input(
                "Bore", min_value=0.1, value=3.2
            )

            stroke = st.number_input(
                "Stroke", min_value=0.1, value=3.4
            )

            compression_ratio = st.number_input(
                "Compression Ratio", min_value=1.0, value=9.0
            )

            horsepower = st.number_input(
                "Horsepower", min_value=1, value=100
            )

            peak_rpm = st.number_input(
                "Peak RPM", min_value=100, value=5000
            )

            city_mpg = st.number_input(
                "City MPG", min_value=1, value=25
            )

            highway_mpg = st.number_input(
                "Highway MPG", min_value=1, value=30
            )

        submitted = st.form_submit_button(
            "🚀 Predict Car Price",
            type="primary",
            use_container_width=True
        )

# =====================================================
# Prediction Section
# =====================================================
with right:
    st.markdown(
        """
        <div class="section-card">
            <div class="section-title">🎯 Prediction Result</div>
            <div class="section-subtitle">
                Your estimated automobile price appears here.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if submitted:
        input_data = {
            "symboling": symboling,
            "normalized-losses": normalized_losses,
            "make": make,
            "fuel-type": fuel_type,
            "aspiration": aspiration,
            "num-of-doors": num_doors,
            "body-style": body_style,
            "drive-wheels": drive_wheels,
            "engine-location": engine_location,
            "wheel-base": wheel_base,
            "length": length,
            "width": width,
            "height": height,
            "curb-weight": curb_weight,
            "engine-type": engine_type,
            "num-of-cylinders": num_cylinders,
            "engine-size": engine_size,
            "fuel-system": fuel_system,
            "bore": bore,
            "stroke": stroke,
            "compression-ratio": compression_ratio,
            "horsepower": horsepower,
            "peak-rpm": peak_rpm,
            "city-mpg": city_mpg,
            "highway-mpg": highway_mpg
        }

        try:
            # Ensure input columns match the training feature names.
            input_df = pd.DataFrame([input_data])
            input_df = input_df.reindex(columns=features)

            # Apply the same cylinder conversion used in training,
            # if the model was trained with numeric cylinder counts.
            if "num-of-cylinders" in input_df.columns:
                cylinder_map = {
                    "two": 2, "three": 3, "four": 4,
                    "five": 5, "six": 6, "eight": 8,
                    "twelve": 12
                }

                input_df["num-of-cylinders"] = (
                    input_df["num-of-cylinders"].replace(cylinder_map)
                )

            # Convert remaining categorical columns to numeric codes
            # only if the training data used the same mapping.
            categorical_columns = input_df.select_dtypes(
                include=["object"]
            ).columns.tolist()

            if categorical_columns:
                st.error(
                    "The model expects numeric features, but these "
                    "inputs are categorical: "
                    + ", ".join(categorical_columns)
                    + ". Save and use the exact preprocessing pipeline "
                    "from training before making predictions."
                )
            else:
                prediction = float(model.predict(input_df)[0])

                st.success("Prediction completed!")

                st.markdown(
                    f"""
                    <div class="price-card">
                        <div class="price-label">
                            Estimated Car Price
                        </div>
                        <div class="price-value">
                            {prediction:,.2f}
                        </div>
                        <p>Estimated price in dataset currency units</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.caption(
                    "This estimate depends on the quality of the "
                    "training data and is not a guaranteed market price."
                )

                with st.expander("📋 View Input Details"):
                    st.json(input_data)

        except Exception as e:
            st.error("Prediction failed. Check your model and features.")
            st.exception(e)

    else:
        st.markdown(
            """
            <div class="placeholder">
                <div style="font-size:45px">🚘</div>
                <h3>Ready for Prediction</h3>
                <p>
                    Enter the vehicle specifications and click
                    <b>Predict Car Price</b> to estimate its price.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

# =====================================================
# Footer
# =====================================================
st.markdown(
    """
    <div style="text-align:center;color:#94a3b8;padding:24px;border-top:1px solid #29364a;margin-top:28px">
        <b style="color:#f8fafc">AUTO VALUE AI</b><br>
        Machine Learning • Streamlit • Decision Tree Regressor
    </div>
    """,
    unsafe_allow_html=True
)