import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
# Set the page layout to wide mode and add a title with emoji
st.set_page_config(
    page_title="💰 Loan Prediction System",
    page_icon="💰",
    layout="wide"
)

# ============================================================================
# LOAD THE TRAINED MODEL
# ============================================================================
# Load the pre-trained model from joblib file
@st.cache_resource  # Cache the model to avoid reloading on every interaction
def load_model():
    """Load the trained loan prediction model"""
    try:
        model = joblib.load('best_loan_model.joblib')
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None

# Load the model
model = load_model()

# ============================================================================
# HEADER SECTION
# ============================================================================
# Create an attractive header with emojis
st.title("💰 Loan Approval Prediction System 🏦")
st.markdown("### 📊 Get instant loan approval predictions based on your profile")
st.markdown("---")

# ============================================================================
# SIDEBAR - USER INPUTS
# ============================================================================
# Create a sidebar for all user inputs
st.sidebar.header("👤 Enter Your Details")
st.sidebar.markdown("Fill in the information below:")

# Gender input with radio buttons
gender = st.sidebar.radio(
    "👫 Gender",
    options=["Male", "Female"],
    help="Select your gender"
)

# Married status input
married = st.sidebar.radio(
    "💑 Marital Status",
    options=["Yes", "No"],
    help="Are you married?"
)

# Number of dependents input
dependents = st.sidebar.selectbox(
    "👨‍👩‍👧‍👦 Number of Dependents",
    options=["0", "1", "2", "3+"],
    help="Number of people dependent on you"
)

# Education level input
education = st.sidebar.radio(
    "🎓 Education Level",
    options=["Graduate", "Not Graduate"],
    help="Your highest education qualification"
)

# Self-employment status input
self_employed = st.sidebar.radio(
    "💼 Employment Type",
    options=["No", "Yes"],
    help="Are you self-employed?"
)

# Loan amount input with number input
loan_amount = st.sidebar.number_input(
    "💵 Loan Amount (in thousands)",
    min_value=0.0,
    max_value=10000.0,
    value=150.0,
    step=10.0,
    help="Enter the loan amount you need (in thousands)"
)

# Loan term input
loan_amount_term = st.sidebar.number_input(
    "📅 Loan Term (in months)",
    min_value=12,
    max_value=480,
    value=360,
    step=12,
    help="Duration of the loan in months"
)

# Credit history input
credit_history = st.sidebar.radio(
    "📜 Credit History",
    options=["1.0 (Good)", "0.0 (Bad)"],
    help="Do you have a good credit history?"
)
# Extract the numeric value from the selection
credit_history_value = float(credit_history.split()[0])

# Property area input
property_area = st.sidebar.selectbox(
    "🏘️ Property Area",
    options=["Urban", "Semiurban", "Rural"],
    help="Location of the property"
)

# Total income input (combined applicant + co-applicant income)
total_income = st.sidebar.number_input(
    "💰 Total Income (Applicant + Co-applicant)",
    min_value=0.0,
    max_value=100000.0,
    value=5000.0,
    step=500.0,
    help="Combined monthly income of applicant and co-applicant"
)

# ============================================================================
# MAIN SECTION - DISPLAY INPUTS AND CALCULATE DERIVED FEATURES
# ============================================================================
# Create columns for better layout
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📋 Your Application Summary")
    
    # Display the input data in a neat format
    st.markdown(f"""
    **Personal Details:**
    - 👫 Gender: {gender}
    - 💑 Married: {married}
    - 👨‍👩‍👧‍👦 Dependents: {dependents}
    - 🎓 Education: {education}
    - 💼 Self Employed: {self_employed}
    
    **Financial Details:**
    - 💵 Loan Amount: ₹{loan_amount * 1000:,.0f}
    - 📅 Loan Term: {loan_amount_term} months
    - 📜 Credit History: {credit_history}
    - 🏘️ Property Area: {property_area}
    - 💰 Total Income: ₹{total_income:,.0f}
    """)

# ============================================================================
# CALCULATE DERIVED FEATURES AT BACKEND
# ============================================================================
# Calculate EMI (Equated Monthly Installment)
# EMI = Loan Amount / Loan Term
emi = loan_amount / loan_amount_term

# Calculate Loan to Income Ratio
# Loan to Income Ratio = Loan Amount / Total Income
loan_to_income_ratio = loan_amount / total_income if total_income > 0 else 0

with col2:
    st.header("🔢 Calculated Metrics")
    
    # Display calculated features in metric cards
    st.metric(
        label="💳 EMI (Monthly)",
        value=f"₹{emi * 1000:.2f}",
        help="Equated Monthly Installment"
    )
    
    st.metric(
        label="📊 Loan-to-Income Ratio",
        value=f"{loan_to_income_ratio:.2f}",
        help="Ratio of loan amount to total income"
    )

st.markdown("---")

# ============================================================================
# ENCODING CATEGORICAL VARIABLES (NO SAVED ENCODERS)
# ============================================================================
# Since no encoders were saved, we manually encode based on common patterns

# Encode Gender: Male=1, Female=0
gender_encoded = 1 if gender == "Male" else 0

# Encode Married: Yes=1, No=0
married_encoded = 1 if married == "Yes" else 0

# Encode Dependents: Convert to numeric
dependents_map = {"0": 0, "1": 1, "2": 2, "3+": 3}
dependents_encoded = dependents_map[dependents]

# Encode Education: Graduate=1, Not Graduate=0
education_encoded = 1 if education == "Graduate" else 0

# Encode Self_Employed: Yes=1, No=0
self_employed_encoded = 1 if self_employed == "Yes" else 0

# ============================================================================
# ONE-HOT ENCODE PROPERTY AREA
# ============================================================================
# The model expects Property_Area as one-hot encoded columns
# Property_Area_Rural, Property_Area_Semiurban, Property_Area_Urban
property_area_rural = 1 if property_area == "Rural" else 0
property_area_semiurban = 1 if property_area == "Semiurban" else 0
property_area_urban = 1 if property_area == "Urban" else 0

# ============================================================================
# CREATE FEATURE DATAFRAME FOR PREDICTION
# ============================================================================
# Create a DataFrame with all features in the EXACT order expected by the model
# Based on the error message, the model expects these exact column names (without Loan_Status):
# Gender, Married, Dependents, Education, Self_Employed, LoanAmount, 
# Loan_Amount_Term, Credit_History, Property_Area_Rural, 
# Property_Area_Semiurban, Property_Area_Urban, Total_income, EMI, Loan_to_Income_Ratio

input_data = pd.DataFrame({
    'Gender': [gender_encoded],
    'Married': [married_encoded],
    'Dependents': [dependents_encoded],
    'Education': [education_encoded],
    'Self_Employed': [self_employed_encoded],
    'LoanAmount': [loan_amount],  # Original value, not log-transformed
    'Loan_Amount_Term': [loan_amount_term],
    'Credit_History': [credit_history_value],
    'Property_Area_Rural': [property_area_rural],
    'Property_Area_Semiurban': [property_area_semiurban],
    'Property_Area_Urban': [property_area_urban],
    'Total_income': [total_income],  # Original value, not log-transformed
    'EMI': [emi],  # Original value, not log-transformed
    'Loan_to_Income_Ratio': [loan_to_income_ratio]  # Original value, not log-transformed
})

# ============================================================================
# PREDICTION SECTION
# ============================================================================
# Create a centered button for prediction
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    # Predict button
    predict_button = st.button("🔮 Predict Loan Status", type="primary", use_container_width=True)

# When the predict button is clicked
if predict_button:
    if model is not None:
        # Show a spinner while making prediction
        with st.spinner("🔄 Processing your application..."):
            try:
                # Make prediction using the loaded model
                prediction = model.predict(input_data)[0]
                
                # Get prediction probability for confidence score
                prediction_proba = model.predict_proba(input_data)[0]
                confidence = max(prediction_proba) * 100
                
                # Display result in a large, prominent box
                st.markdown("---")
                st.header("🎯 Prediction Result")
                
                # Show result based on prediction
                if prediction == 1:  # Loan Approved
                    st.success("### ✅ Congratulations! Your Loan is APPROVED! 🎉")
                    st.balloons()  # Show balloons animation
                    st.markdown(f"""
                    <div style='background-color: #d4edda; padding: 20px; border-radius: 10px; border-left: 5px solid #28a745;'>
                        <h3 style='color: #155724;'>✅ Loan Approved</h3>
                        <p style='font-size: 18px; color: #155724;'>
                            Based on your profile, the system predicts that your loan application 
                            will be <strong>APPROVED</strong>!
                        </p>
                        <p style='font-size: 16px; color: #155724;'>
                            📊 Confidence Score: <strong>{confidence:.2f}%</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:  # Loan Rejected
                    st.error("### ❌ Sorry, Your Loan is REJECTED 😔")
                    st.markdown(f"""
                    <div style='background-color: #f8d7da; padding: 20px; border-radius: 10px; border-left: 5px solid #dc3545;'>
                        <h3 style='color: #721c24;'>❌ Loan Rejected</h3>
                        <p style='font-size: 18px; color: #721c24;'>
                            Based on your profile, the system predicts that your loan application 
                            might be <strong>REJECTED</strong>.
                        </p>
                        <p style='font-size: 16px; color: #721c24;'>
                            📊 Confidence Score: <strong>{confidence:.2f}%</strong>
                        </p>
                        <p style='font-size: 14px; color: #721c24; margin-top: 15px;'>
                            💡 <strong>Suggestions:</strong><br>
                            • Improve your credit history<br>
                            • Reduce the loan amount or increase loan term<br>
                            • Increase your total income<br>
                            • Consider adding a co-applicant
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
            except Exception as e:
                # Display error if prediction fails
                st.error(f"❌ Error making prediction: {str(e)}")
                st.info("💡 Please check if all input values are valid and try again.")
                
                # Debug information
                with st.expander("🔍 Debug Information"):
                    st.write("Input Data:")
                    st.dataframe(input_data)
                    st.write("Data Types:")
                    st.write(input_data.dtypes)
    else:
        # Display error if model is not loaded
        st.error("❌ Model not loaded. Please check if 'best_loan_model.joblib' exists.")

# ============================================================================
# FOOTER SECTION
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>💡 <strong>Note:</strong> This is a prediction model and results are probabilistic.</p>
    <p>📞 For actual loan approval, please contact your bank or financial institution.</p>
    <p style='margin-top: 20px; font-size: 12px;'>
        Made with ❤️ using Streamlit | 🤖 Powered by Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)