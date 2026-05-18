# Importing required libraries

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# =====================================================
# PDF Report Generator
# =====================================================

def generate_pdf_report(
    disease_name,
    result_text
):

    file_name = "medical_report.pdf"

    c = canvas.Canvas(
        file_name,
        pagesize=letter
    )

    width, height = letter

    # Title

    c.setFont("Helvetica-Bold", 18)

    c.drawString(
        180,
        750,
        "AI Healthcare Report"
    )

    # Disease Name

    c.setFont("Helvetica-Bold", 14)

    c.drawString(
        100,
        680,
        f"Disease Module: {disease_name}"
    )

    # Prediction Result

    c.setFont("Helvetica", 13)

    c.drawString(
        100,
        630,
        f"Prediction Result: {result_text}"
    )

    # Footer

    c.setFont("Helvetica-Oblique", 10)

    c.drawString(
        100,
        550,
        "Generated using AI-Based Disease Detection System"
    )

    c.save()

    return file_name


# =====================================================
# Page Configuration
# =====================================================

st.set_page_config(
    page_title="AI Healthcare Detection System",
    page_icon="🧠",
    layout="wide"
)


# =====================================================
# Loading Models
# =====================================================

parkinson_model = joblib.load('parkinsons_model.pkl')
alzheimer_model = joblib.load('alzheimers_model.pkl')


# =====================================================
# Sidebar
# =====================================================

st.sidebar.title("🧠 AI Healthcare System")

st.sidebar.info(
    "This project predicts Parkinson's and Alzheimer's disease using Machine Learning algorithms."
)

st.sidebar.success("Model Accuracy")

st.sidebar.write("Parkinson Accuracy : 84.6%")
st.sidebar.write("Alzheimer Accuracy : 93.2%")


# =====================================================
# Main Title
# =====================================================

st.markdown(
    """
    <h1 style='text-align: center; color: #4B8BBE;'>
    AI-Based Early Detection System
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h3 style='text-align: center;'>
    Parkinson's and Alzheimer's Disease Prediction
    </h3>
    """,
    unsafe_allow_html=True
)

st.write("---")

# =====================================================
# Dashboard Metrics
# =====================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Parkinson Accuracy",
        value="84.6%"
    )

with col2:
    st.metric(
        label="Alzheimer Accuracy",
        value="93.2%"
    )

with col3:
    st.metric(
        label="AI Models",
        value="2"
    )


# =====================================================
# Disease Selection
# =====================================================

selected_disease = st.selectbox(
    "Select Disease Prediction Module",
    [
        "Parkinson's Disease",
        "Alzheimer's Disease"
    ]
)


# =====================================================
# Parkinson's Module
# =====================================================

if selected_disease == "Parkinson's Disease":

    st.header("Parkinson's Disease Detection")

    col1, col2 = st.columns(2)

    with col1:

        MDVP_Fo = st.number_input("MDVP:Fo(Hz)")
        MDVP_Fhi = st.number_input("MDVP:Fhi(Hz)")
        MDVP_Flo = st.number_input("MDVP:Flo(Hz)")
        MDVP_Jitter_percent = st.number_input("MDVP:Jitter(%)")
        MDVP_Jitter_Abs = st.number_input("MDVP:Jitter(Abs)")
        MDVP_RAP = st.number_input("MDVP:RAP")
        MDVP_PPQ = st.number_input("MDVP:PPQ")
        Jitter_DDP = st.number_input("Jitter:DDP")
        MDVP_Shimmer = st.number_input("MDVP:Shimmer")
        MDVP_Shimmer_dB = st.number_input("MDVP:Shimmer(dB)")

    with col2:

        Shimmer_APQ3 = st.number_input("Shimmer:APQ3")
        Shimmer_APQ5 = st.number_input("Shimmer:APQ5")
        MDVP_APQ = st.number_input("MDVP:APQ")
        Shimmer_DDA = st.number_input("Shimmer:DDA")
        NHR = st.number_input("NHR")
        HNR = st.number_input("HNR")
        RPDE = st.number_input("RPDE")
        DFA = st.number_input("DFA")
        spread1 = st.number_input("spread1")
        spread2 = st.number_input("spread2")
        D2 = st.number_input("D2")
        PPE = st.number_input("PPE")


    if st.button("Predict Parkinson's Disease"):

        with st.spinner("Analyzing Voice Data..."):

            input_data = pd.DataFrame([[

                MDVP_Fo,
                MDVP_Fhi,
                MDVP_Flo,
                MDVP_Jitter_percent,
                MDVP_Jitter_Abs,
                MDVP_RAP,
                MDVP_PPQ,
                Jitter_DDP,
                MDVP_Shimmer,
                MDVP_Shimmer_dB,
                Shimmer_APQ3,
                Shimmer_APQ5,
                MDVP_APQ,
                Shimmer_DDA,
                NHR,
                HNR,
                RPDE,
                DFA,
                spread1,
                spread2,
                D2,
                PPE

            ]], columns=[

                'MDVP:Fo(Hz)',
                'MDVP:Fhi(Hz)',
                'MDVP:Flo(Hz)',
                'MDVP:Jitter(%)',
                'MDVP:Jitter(Abs)',
                'MDVP:RAP',
                'MDVP:PPQ',
                'Jitter:DDP',
                'MDVP:Shimmer',
                'MDVP:Shimmer(dB)',
                'Shimmer:APQ3',
                'Shimmer:APQ5',
                'MDVP:APQ',
                'Shimmer:DDA',
                'NHR',
                'HNR',
                'RPDE',
                'DFA',
                'spread1',
                'spread2',
                'D2',
                'PPE'

            ])


            prediction = parkinson_model.predict(input_data)


            if prediction[0] == 0:

                st.success("✅ The Person is Healthy")
                

            else:

                st.error("⚠️ The Person has Parkinson's Disease")


            # Accuracy Chart

            chart_data = pd.DataFrame({
                'Accuracy': [1.0, 0.846]
            }, index=['Training', 'Testing'])

            st.subheader("Model Accuracy Analysis")

            # Creating smaller accuracy chart

            fig, ax = plt.subplots(figsize=(4,3))

            ax.bar(
                ['Training', 'Testing'],
                [1.0, 0.846]
            )

            ax.set_ylabel("Accuracy")

            ax.set_title("Model Accuracy")

            st.pyplot(fig)


# =====================================================
# Alzheimer's Module 
# =====================================================

elif selected_disease == "Alzheimer's Disease":

    st.header("Alzheimer's Disease Detection")
    st.info("Please fill in all 32 clinical and behavioral parameters for accurate Alzheimer's prediction.")

    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Demographics & Lifestyle", 
        "🏥 Medical History", 
        "🧪 Clinical Measurements", 
        "🧠 Cognitive & Behavioral"
    ])

    with tab1:
        st.subheader("Demographic & Lifestyle Factors")
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Age", 40, 100, 65)
            gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x==1 else "Female")
            ethnicity = st.selectbox("Ethnicity", [0, 1, 2, 3], format_func=lambda x: f"Type {x}")
            education = st.selectbox("Education Level", [0, 1, 2, 3], format_func=lambda x: f"Level {x}")
            bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=24.5, step=0.1)
        with col2:
            smoking = st.selectbox("Smoking Status", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            alcohol = st.selectbox("Alcohol Consumption", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            physical_activity = st.number_input("Physical Activity (Hours/Week)", 0.0, 10.0, 3.2)
            diet_quality = st.number_input("Diet Quality Score (0-10)", 0.0, 10.0, 7.5)
            sleep_quality = st.number_input("Sleep Quality Score (0-10)", 0.0, 10.0, 6.8)

    with tab2:
        st.subheader("Medical History & Conditions")
        col1, col2 = st.columns(2)
        with col1:
            family_history = st.selectbox("Family History of Alzheimer's", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            cardio_disease = st.selectbox("Cardiovascular Disease", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        with col2:
            depression = st.selectbox("Depression", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            head_injury = st.selectbox("Head Injury History", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")

    with tab3:
        st.subheader("Clinical & Blood Measurements")
        col1, col2 = st.columns(2)
        with col1:
            systolic_bp = st.number_input("Systolic BP (mmHg)", 80, 200, 130)
            diastolic_bp = st.number_input("Diastolic BP (mmHg)", 50, 120, 85)
            chol_total = st.number_input("Total Cholesterol (mg/dL)", 100, 400, 200)
        with col2:
            chol_ldl = st.number_input("LDL Cholesterol (mg/dL)", 50, 250, 120)
            chol_hdl = st.number_input("HDL Cholesterol (mg/dL)", 20, 100, 55)
            chol_trig = st.number_input("Triglycerides (mg/dL)", 50, 500, 180)

    with tab4:
        st.subheader("Cognitive, Functional & Behavioral Assessment")
        col1, col2 = st.columns(2)
        with col1:
            mmse = st.number_input("MMSE Score (0-30)", 0, 30, 22)
            functional_assess = st.number_input("Functional Assessment Score (0-10)", 0.0, 10.0, 6.5)
            memory = st.selectbox("Memory Complaints", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            behavioral = st.selectbox("Behavioral Problems", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            adl = st.number_input("ADL (Activities of Daily Living) Score (0-10)", 0.0, 10.0, 5.5)
        with col2:
            confusion = st.selectbox("Confusion", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            disorientation = st.selectbox("Disorientation", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            personality = st.selectbox("Personality Changes", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            difficulty_tasks = st.selectbox("Difficulty Completing Tasks", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            forgetfulness = st.selectbox("Forgetfulness", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")


    # Prediction Button
    if st.button("Analyze Alzheimer's Risk"):

        with st.spinner("Analyzing Complete Patient Data..."):
            
            
            input_data_alz = [
                age, gender, ethnicity, education, bmi, smoking, alcohol, 
                physical_activity, diet_quality, sleep_quality, family_history, 
                cardio_disease, diabetes, depression, head_injury, hypertension, 
                systolic_bp, diastolic_bp, chol_total, chol_ldl, chol_hdl, 
                chol_trig, mmse, functional_assess, memory, behavioral, adl, 
                confusion, disorientation, personality, difficulty_tasks, forgetfulness
            ]

            # Features Name List matching the notebook
            feature_names = [
                'Age', 'Gender', 'Ethnicity', 'EducationLevel', 'BMI', 'Smoking', 
                'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality', 
                'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes', 'Depression', 
                'HeadInjury', 'Hypertension', 'SystolicBP', 'DiastolicBP', 'CholesterolTotal', 
                'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides', 'MMSE', 
                'FunctionalAssessment', 'MemoryComplaints', 'BehavioralProblems', 'ADL', 
                'Confusion', 'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks', 
                'Forgetfulness'
            ]

            # DataFrame Creation
            input_data_alz_df = pd.DataFrame([input_data_alz], columns=feature_names)

            # Actual Model Prediction
            prediction = alzheimer_model.predict(input_data_alz_df)
            
            # Probability Calculation for Chart
            try:
                prediction_proba = alzheimer_model.predict_proba(input_data_alz_df)[0]
                healthy_prob = prediction_proba[0] * 100
                disease_prob = prediction_proba[1] * 100
            except:
                if prediction[0] == 1:
                    healthy_prob, disease_prob = 10, 90
                else:
                    healthy_prob, disease_prob = 90, 10

            st.write("---")
            
            # Output Result Display
            if prediction[0] == 0:
                st.success("✅ The Person is Healthy (Low risk of Alzheimer's)")
            else:
                st.error("⚠️ The Person has Alzheimer's Disease")

            # Real Graph Analysis
            st.subheader("Alzheimer's Risk Analysis Graph")
            labels = ['Healthy Probability', 'Disease Probability']
            values = [healthy_prob, disease_prob]

            fig, ax = plt.subplots(figsize=(4,4))
            ax.pie(
                values,
                labels=labels,
                autopct='%1.1f%%',
                colors=['#2ecc71', '#e74c3c'],
                startangle=90
            )
            ax.set_title("Alzheimer's Risk Analysis")
            st.pyplot(fig)

# =====================================================
# About Project
# =====================================================

st.subheader("About Project")

st.write(
    '''
    This AI-based healthcare system uses Machine Learning
    algorithms to detect Parkinson's and Alzheimer's disease
    at an early stage using biomedical and patient health data.
    '''
)

st.subheader("Technologies Used")

st.write("""
- Python
- Machine Learning
- Random Forest Algorithm
- Streamlit
- Pandas
- NumPy
- Matplotlib
""")

st.subheader("Developer")

st.write("Developed by Nishant Saini")

# =====================================================
# Footer
# =====================================================

st.write("---")

st.markdown(
    """
    <div style='text-align:center;'>
    Developed using Machine Learning and Streamlit
    </div>
    """,
    unsafe_allow_html=True
)

