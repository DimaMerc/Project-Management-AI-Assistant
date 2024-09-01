import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ai_project_management import ProjectManagementAI 
import sys
import io
import traceback

# Initialize the AI
@st.cache_resource
def load_ai():
    pm_ai = ProjectManagementAI()
    data = pm_ai.generate_synthetic_data(5000)
    X = pm_ai.preprocess_data(data)
    y_risk = data['risk_level']
    y_effort = data['duration']
    y_team = pd.get_dummies(data['team_size'])
    y_pattern = data['success']

    pm_ai.train_risk_classifier(X, y_risk)
    pm_ai.train_effort_estimator(X, y_effort)
    pm_ai.train_team_generator(X.values, y_team.values)

    pattern_data = np.array([X.values[i:i+10] for i in range(len(X)-10)])
    pattern_target = y_pattern[10:].values

    pm_ai.train_pattern_recognizer(pattern_data, pattern_target)
    return pm_ai, X

pm_ai, X = load_ai()

st.title('Project Management AI Assistant')

# Input form for project details
st.header('Enter Project Details')
duration = st.slider('Estimated Duration (days)', 30, 365, 180)
budget = st.number_input('Budget ($)', 10000, 1000000, 500000)
team_size = st.slider('Team Size', 3, 20, 10)
complexity = st.selectbox('Project Complexity', ['Low', 'Medium', 'High'])
industry = st.selectbox('Industry', ['Tech', 'Finance', 'Healthcare', 'Retail'])
project_type = st.selectbox('Project Type', ['Software', 'Hardware', 'Service', 'Research'])
manager_experience = st.slider('Manager Experience (years)', 1, 20, 5)

# Create a new project
new_project = pd.DataFrame({
    'duration': [duration],
    'budget': [budget],
    'team_size': [team_size],
    'complexity': [complexity],
    'industry': [industry],
    'project_type': [project_type],
    'manager_experience': [manager_experience]
})

# Preprocess the new project data
new_project_processed = pm_ai.preprocess_data(new_project, fit=False)

# Team Sentiment Analysis
st.header('Team Sentiment Analysis')
uploaded_file = st.file_uploader("Upload a team photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    team_sentiment = pm_ai.analyze_team_sentiment(uploaded_file)
    st.write("Team Sentiment Analysis:", team_sentiment)

    # Update project risk based on team sentiment
    updated_risk = pm_ai.update_project_risk(new_project_processed, team_sentiment)
    st.write("Updated Risk Assessment:", updated_risk)
else:
    st.write("Please upload an image to analyze team sentiment.")
    team_sentiment = None

if st.button('Analyze Project'):
    try:
        # Risk Classification
        st.header('Risk Analysis')
        risk_probabilities, risk_score = pm_ai.classify_risk(new_project_processed)
        for risk, prob in risk_probabilities.items():
            st.write(f"{risk} risk: {prob:.2%}")
        st.write(f"Overall risk score: {risk_score:.2f}")

        # Update risk assessment based on team sentiment
        if team_sentiment:
            updated_risk = pm_ai.update_project_risk(new_project_processed, team_sentiment)
            st.write("Updated Risk Assessment based on Team Sentiment:")
            for risk, prob in updated_risk.items():
                st.write(f"{risk} risk: {prob:.2%}")

        # Effort Estimation
        st.header('Effort Estimation')
        effort_estimate = pm_ai.estimate_effort(new_project_processed)
        st.write(f"Estimated project duration: {effort_estimate:.0f} days")
        if effort_estimate > duration:
            st.write("Note: The estimated duration is significantly longer than initially planned. Consider reviewing project scope and resources.")

        # Team Composition
        st.header('Recommended Team Composition')
        team_composition = pm_ai.generate_team(new_project)
        fig, ax = plt.subplots()
        roles = list(team_composition.keys())
        percentages = list(team_composition.values())
        ax.bar(roles, percentages)
        plt.xticks(rotation=45, ha='right')
        plt.title('Team Composition')
        plt.ylabel('Percentage')
        st.pyplot(fig)

        for role, percentage in team_composition.items():
            st.write(f"{role}: {percentage:.2%}")

        # Success Prediction
        st.header('Success Prediction')
        project_data = new_project_processed.values
        print(f"Project data shape: {project_data.shape}")  # Debug print
        success_probability, success_explanation = pm_ai.predict_success(project_data, risk_score)
        st.write(f"Predicted success probability: {success_probability:.2%}")
        st.write("Explanation:")
        st.write(success_explanation)


        # Project Simulation
        st.header('Project Simulation')
        project_simulation = pm_ai.simulate_project(new_project_processed.values[0], risk_score, success_probability)

        plt.figure(figsize=(10, 5))
        plt.plot(project_simulation)
        plt.title('Project Progress Simulation')
        plt.xlabel('Time Steps')
        plt.ylabel('Progress')
        plt.ylim(0, 1)
        st.pyplot(plt)

        st.write("Simulation Summary:")
        st.write(f"Initial progress: {project_simulation[0]:.2%}")
        st.write(f"Final progress: {project_simulation[-1]:.2%}")
        st.write(f"Progress change: {project_simulation[-1] - project_simulation[0]:.2%}")
        st.write(f"Number of steps: {len(project_simulation)}")

    except Exception as e:
        st.write(f"An error occurred during project analysis: {str(e)}")
        st.write("Error details:")
        st.write(traceback.format_exc())

        st.write("Debug information:")
        st.write(f"new_project_processed shape: {new_project_processed.shape}")
        st.write(f"new_project_processed dtypes: {new_project_processed.dtypes}")
        st.write(f"new_project_processed values:")
        st.write(new_project_processed)

st.sidebar.header('About')
st.sidebar.write("""
This Project Management AI Assistant uses machine learning to provide insights on project risk, 
effort estimation, team composition, and success prediction. It also includes team sentiment analysis 
based on uploaded photos. It's trained on synthetic data and is meant for demonstration purposes only.
""")