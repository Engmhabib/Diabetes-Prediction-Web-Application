import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('cleaned_diabetes_data.csv')
    return data

# Function to load the trained model
@st.cache_resource
def load_model():
    model = joblib.load('random_forest_diabetes_model.joblib')
    return model

# Function to get user input
def user_input_features():
    age = st.sidebar.slider('Age', 0, 100, 50)
    hypertension = st.sidebar.selectbox('Hypertension', (0, 1))
    heart_disease = st.sidebar.selectbox('Heart Disease', (0, 1))
    bmi = st.sidebar.slider('BMI', 10.0, 50.0, 25.0)
    hba1c_level = st.sidebar.slider('HbA1c Level', 3.5, 9.0, 5.5)
    blood_glucose_level = st.sidebar.slider('Blood Glucose Level', 80.0, 300.0, 120.0)
    gender = st.sidebar.selectbox('Gender', ('Female', 'Male', 'Other'))
    smoking_status = st.sidebar.selectbox('Smoking History', 
                                          ('current', 'ever', 'former', 'never', 'not current'))

    # One-hot encoding for categorical variables
    gender_features = pd.get_dummies(pd.Series(gender), prefix='gender')
    smoking_features = pd.get_dummies(pd.Series(smoking_status), prefix='smoking_history')

    # Create a data frame of the input features
    data = {
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'bmi': bmi,
        'HbA1c_level': hba1c_level,
        'blood_glucose_level': blood_glucose_level,
    }

    features = pd.DataFrame(data, index=[0])
    features = pd.concat([features, gender_features, smoking_features], axis=1)

    # Ensure all expected columns are present
    expected_cols = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level',
                     'blood_glucose_level', 'gender_Female', 'gender_Male', 'gender_Other',
                     'smoking_history_current', 'smoking_history_ever', 'smoking_history_former',
                     'smoking_history_never', 'smoking_history_not current']
    for col in expected_cols:
        if col not in features.columns:
            features[col] = 0

    return features

# Main
def main():
    st.title("Diabetes Prediction App")

    # Load data & model
    data = load_data()
    model = load_model()

    # Sidebar for navigation
    st.sidebar.subheader("Navigation")
    options = st.sidebar.radio("Select Page:", ["Data Visualization", "Make Prediction"])

    # Data Visualization Page
    if options == "Data Visualization":
        st.subheader("Dataset Overview")
        st.write(data.describe())

        # Visualizing distribution of age
        st.subheader("Age Distribution")
        plt.figure(figsize=(10, 6))
        sns.histplot(data['age'], kde=True)
        st.pyplot(plt)

        # Visualizing distribution of BMI
        st.subheader("BMI Distribution")
        plt.figure(figsize=(10, 6))
        sns.histplot(data['bmi'], kde=True)
        st.pyplot(plt)

        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        plt.figure(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
        st.pyplot(plt)

    # Prediction Page
    elif options == "Make Prediction":
        st.subheader("Predict Diabetes")

        # User input for prediction
        input_df = user_input_features()

        # Display user input
        st.write('Specified Input features')
        st.write(input_df)

        # Prediction
        if st.button("Predict"):
            prediction = model.predict(input_df)
            st.write(f"Prediction: {'Diabetic' if prediction[0] == 1 else 'Not Diabetic'}")

# Run the app
if __name__ == '__main__':
    main()
