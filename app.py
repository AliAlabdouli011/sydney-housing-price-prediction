import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Sydney Housing Price Predictor", page_icon="house")

model = joblib.load("model.joblib")
model_columns = joblib.load("model_columns.joblib")
suburb_options = joblib.load("suburb_options.joblib")
property_type_options = joblib.load("property_type_options.joblib")

st.title("Sydney Housing Price Predictor")
st.write(
    "Estimate a sale price for a property in Mosman, Parramatta, or Campbelltown, "
    "based on a Decision Tree model trained on 104 reported sold listings (83 used "
    "for training after the 80:20 train/test split). "
    "Model, columns, and category options are loaded from model.joblib, "
    "model_columns.joblib, suburb_options.joblib, and property_type_options.joblib, "
    "which must be present alongside this file."
)

st.warning(
    "This is a point estimate, not a valuation. It is least reliable for Mosman "
    "properties and for any combination of features uncommon in the training data "
    "(only 104 properties collected: 21 Mosman, 28 Parramatta, and 34 Campbelltown properties "
    "in the training split)."
)

st.header("Property details")

col1, col2 = st.columns(2)

with col1:
    suburb = st.selectbox("Suburb", suburb_options)
    property_type = st.selectbox("Property type", property_type_options)
    beds = st.number_input("Bedrooms", min_value=1, max_value=6, value=3,
                            help="Training data range: 1-6")
    baths = st.number_input("Bathrooms", min_value=1, max_value=4, value=2,
                             help="Training data range: 1-4")

with col2:
    parking = st.number_input("Parking spaces", min_value=0, max_value=8, value=1,
                               help="Training data range: 0-8")
    land_size_known = st.radio(
        "Land size",
        ["Unknown / not applicable (typical for apartments/units on strata title)", "Known"],
        index=0,
    )
    if land_size_known == "Known":
        land_size = st.number_input("Land size (sqm)", min_value=74, max_value=1233, value=300,
                                     help="Training data range where recorded: 74-1233 sqm")
    else:
        land_size = 0
    sale_year = st.number_input("Sale year", min_value=2017, max_value=2026, value=2026,
                                 help="Training data range: 2017-2026. Years outside this range "
                                      "are extrapolation the model was not trained to handle.")

if st.button("Predict price", type="primary"):
    has_land = land_size_known == "Known"

    input_row = {col: 0 for col in model_columns}
    input_row["beds"] = beds
    input_row["baths"] = baths
    input_row["parking"] = parking
    input_row["has_land_size"] = int(has_land)
    input_row["land_size_filled"] = land_size if has_land else 0
    input_row["sale_year"] = sale_year

    suburb_col = f"suburb_{suburb}"
    if suburb_col in input_row:
        input_row[suburb_col] = 1

    type_col = f"property_type_{property_type}"
    if type_col in input_row:
        input_row[type_col] = 1

    input_df = pd.DataFrame([input_row])[model_columns]
    prediction = model.predict(input_df)[0]

    st.header("Predicted price")
    st.success(f"${prediction:,.0f}")

    if suburb == "Mosman":
        st.caption(
            "Mosman properties showed the largest prediction errors in testing "
            "(see report Part 4), treat this estimate with extra caution."
        )
    st.caption(
        "This is a point estimate from a single Decision Tree (max_depth=5), which produces "
        "stepwise predictions: structurally different properties can receive the exact same "
        "estimate if they fall into the same region of the tree. It does not know about "
        "renovation status, views, or precinct-level location, factors likely to explain "
        "some of its largest errors. See the accompanying report for full known limitations."
    )

st.divider()
st.caption(
    "Model: Decision Tree Regressor (max_depth=5), trained on 83 of 104 collected "
    "properties, Test R2=0.85. A single tree is fast to train and deploy but, with only "
    "83 training rows, is less stable than an ensemble method would be, small changes "
    "in the training data could shift its splits meaningfully. "
    "Built for SIT307/SIT720 8.1 Distinction Task."
)
