import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Create dataset manually (if file is not provided)
data = {'HoursStudied': [1.5, 3.2, 5.0, 7.1, 8.5], 'ExamScore': [30, 50, 75, 85, 95]}
df = pd.DataFrame(data)

# Split data into Features (X) and Target (y)
X = df[['HoursStudied']]  # Independent variable
y = df['ExamScore']  # Dependent variable

# Train-Test Split (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit UI
st.title("📘 Student Exam Score Predictor")
st.write("Enter the number of hours studied to predict the exam score.")

# User Input
hours = st.number_input("Hours Studied:", min_value=0.0, max_value=12.0, step=0.1)

# Predict Button
if st.button("Predict Score"):
    predicted_score = model.predict(pd.DataFrame([[hours]], columns=["HoursStudied"]))[0]
    st.success(f"📈 Predicted Exam Score: {predicted_score:.2f}")

# Show Sample Data
st.write("### 📊 Sample Training Data")
st.dataframe(df)
