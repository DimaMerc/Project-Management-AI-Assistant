import numpy as np
import pandas as pd
import os
from typing import List, Dict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
import clip
import io
from streamlit.runtime.uploaded_file_manager import UploadedFile
import torch.nn.functional as F

class LSTMRegressor(nn.Module):
    def __init__(self, input_size=15, hidden_size=50, output_size=1):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.linear(lstm_out[:, -1, :])

class ProjectManagementAI:
    def __init__(self):
        self.risk_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.effort_estimator = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.team_generator = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 5),
            nn.Softmax(dim=1)
        )
        self.pattern_recognizer = None
        self.scaler = StandardScaler()
        self.feature_names = None  # Initialize as None
        self.label_encoder = LabelEncoder()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)


    def generate_synthetic_data(self, n_samples=1000):
        np.random.seed(42)
        data = {
            'duration': np.random.randint(30, 365, n_samples),
            'budget': np.random.randint(10000, 1000000, n_samples),
            'team_size': np.random.randint(3, 20, n_samples),
            'complexity': np.random.choice(['Low', 'Medium', 'High'], n_samples),
            'risk_level': np.random.choice(['Low', 'Medium', 'High'], n_samples),
            'success': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            'industry': np.random.choice(['Tech', 'Finance', 'Healthcare', 'Retail'], n_samples),
            'project_type': np.random.choice(['Software', 'Hardware', 'Service', 'Research'], n_samples),
            'manager_experience': np.random.randint(1, 20, n_samples)
        }
        df = pd.DataFrame(data)

        # Add some correlations
        df.loc[df['complexity'] == 'High', 'duration'] += 100
        df.loc[df['budget'] > 500000, 'team_size'] += 5
        df.loc[(df['complexity'] == 'High') & (df['team_size'] < 10), 'risk_level'] = 'High'

        return df
        
    def preprocess_data(self, data, fit=True):
        # Define all possible categories for each categorical variable
        all_complexities = ['Low', 'Medium', 'High']
        all_industries = ['Tech', 'Finance', 'Healthcare', 'Retail']
        all_project_types = ['Software', 'Hardware', 'Service', 'Research']

        # Define the expected column order
        expected_columns = ['duration', 'budget', 'team_size', 'manager_experience'] + \
                           [f'complexity_{c}' for c in all_complexities] + \
                           [f'industry_{i}' for i in all_industries] + \
                           [f'project_type_{p}' for p in all_project_types]

        # Check if data is already preprocessed
        if self.feature_names is not None and set(self.feature_names).issubset(set(data.columns)):
            return data  # Data is already preprocessed, return as is

        # One-hot encode categorical variables
        X = pd.get_dummies(data, columns=['complexity', 'industry', 'project_type'])

        # Add missing columns (if any) and ensure consistent order
        for col in expected_columns:
             if col not in X.columns:
                 X[col] = 0

        # Ensure consistent column order
        X = X.reindex(columns=expected_columns, fill_value=0)

        if fit:
            # Fit the scaler on all columns
            self.scaler = StandardScaler()
            self.scaler.fit(X)
            self.feature_names = X.columns.tolist()
        else:
            # Ensure all expected columns are present
            if self.feature_names is not None:
                missing_cols = set(self.feature_names) - set(X.columns)
                for col in missing_cols:
                    X[col] = 0
                X = X[self.feature_names]

        # Transform all columns
        X_scaled = pd.DataFrame(self.scaler.transform(X), columns=X.columns)

        return X_scaled

    def train_risk_classifier(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.risk_classifier.fit(X_train, y_train)
        y_pred = self.risk_classifier.predict(X_test)
        print("Risk Classifier Performance:")
        print(classification_report(y_test, y_pred))

    def classify_risk(self, project_features):
        probabilities = self.risk_classifier.predict_proba(project_features)
        risk_dict = dict(zip(self.risk_classifier.classes_, probabilities[0]))
    
    # Calculate a weighted risk score
        risk_score = risk_dict['High'] * 1 + risk_dict['Medium'] * 0.5 + risk_dict['Low'] * 0.1
    
        return risk_dict, risk_score

  
 


    def predict_success(self, project_data, risk_score):
        with torch.no_grad():
            flattened_data = project_data.flatten()
            input_size = 15  # Adjust if different
            sequence_length = len(flattened_data) // input_size
            input_tensor = torch.FloatTensor(flattened_data).view(1, sequence_length, input_size)
        
            raw_output = self.pattern_recognizer(input_tensor)
        
        # Apply sigmoid to the raw output
            base_success_probability = torch.sigmoid(raw_output).item()
    
        print(f"Raw output from pattern recognizer: {raw_output.item()}")
        print(f"Base success probability after sigmoid: {base_success_probability}")
    
    # Extract and normalize relevant features
        duration = min(project_data[0, 0] / 365, 1)
        budget = min(project_data[0, 1] / 1000000, 1)
        team_size = min(project_data[0, 2] / 20, 1)
        complexity = project_data[0, 4]
        manager_experience = min(project_data[0, 6] / 20, 1)
    
    # Softer risk adjustment
        risk_adjustment = 1 - (risk_score ** 2)  # Square to soften impact of moderate risks
    
    # Softer complexity adjustment
        complexity_adjustment = 1 - (complexity - 0.5) ** 2 * 0.5  # Reduced impact of complexity
    
    # Positive factor adjustments
        team_adjustment = 1 + team_size * 0.3  # Up to 30% boost for large teams
        experience_adjustment = 1 + manager_experience * 0.3  # Up to 30% boost for experienced managers
        budget_adjustment = 1 + budget * 0.2  # Up to 20% boost for high budgets
        duration_adjustment = 1 - duration * 0.1  # Slight negative impact for longer projects
    
    # Combine adjustments with a more balanced approach
        negative_factors = (risk_adjustment * complexity_adjustment * duration_adjustment)
        positive_factors = (team_adjustment * experience_adjustment * budget_adjustment)
        final_adjustment = (negative_factors + positive_factors) / 2
    
    # Calculate final success probability
        final_success_probability = base_success_probability * final_adjustment
    
    # Ensure final probability is between 0 and 1
        final_success_probability = max(0, min(1, final_success_probability))
    
    # Generate explanation
        explanation = f"Raw output from pattern recognizer: {raw_output.item():.4f}\n"
        explanation += f"Base success probability (after sigmoid): {base_success_probability:.2%}\n"
        explanation += f"Risk adjustment (score: {risk_score:.2f}): {risk_adjustment:.2f}\n"
        explanation += f"Complexity adjustment: {complexity_adjustment:.2f}\n"
        explanation += f"Team size adjustment: {team_adjustment:.2f}\n"
        explanation += f"Manager experience adjustment: {experience_adjustment:.2f}\n"
        explanation += f"Budget adjustment: {budget_adjustment:.2f}\n"
        explanation += f"Duration adjustment: {duration_adjustment:.2f}\n"
        explanation += f"Final adjustment: {final_adjustment:.2f}\n"
        explanation += f"Final success probability: {final_success_probability:.2%}"
    
        return final_success_probability, explanation
    
    def train_effort_estimator(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.effort_estimator.fit(X_train, y_train)
        y_pred = self.effort_estimator.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Effort Estimator Mean Squared Error: {mse}")

    def estimate_effort(self, project_features):
        return self.effort_estimator.predict(project_features)[0]

    def train_team_generator(self, X, y, epochs=50, early_stop_patience=5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train_float = self.scaler.fit_transform(X_train.astype(float))
        X_test_float = self.scaler.transform(X_test.astype(float))
        
        X_train = torch.FloatTensor(X_train_float)
        y_train = torch.FloatTensor(y_train.astype(float))
        X_test = torch.FloatTensor(X_test_float)
        y_test = torch.FloatTensor(y_test.astype(float))
        
        input_size = X_train.shape[1]
        self.team_generator = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, y_train.shape[1]),
            nn.Softmax(dim=1)
        )
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.team_generator.parameters(), lr=0.001)
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.team_generator.train()
            outputs = self.team_generator(X_train)
            loss = criterion(outputs, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            self.team_generator.eval()
            with torch.no_grad():
                test_outputs = self.team_generator(X_test)
                test_loss = criterion(test_outputs, y_test)
                accuracy = ((test_outputs.round() == y_test).float().mean())
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Accuracy: {accuracy:.4f}")
            
            if test_loss < best_loss:
                best_loss = test_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print(f"Final Team Generator Accuracy: {accuracy:.4f}")


    def generate_team(self, project_requirements):
    # Ensure project_requirements is a DataFrame
        if not isinstance(project_requirements, pd.DataFrame):
           project_requirements = pd.DataFrame([project_requirements])

    # Preprocess the data consistently with the training data
        preprocessed_requirements = self.preprocess_data(project_requirements, fit=False)

    # Convert to tensor
        input_tensor = torch.FloatTensor(preprocessed_requirements.values)

        with torch.no_grad():
            team_composition = self.team_generator(input_tensor)
        # Apply softmax to ensure non-negative values that sum to 1
            team_composition = F.softmax(team_composition, dim=1)

        roles = ['Project Manager', 'Developer', 'Designer', 'Tester', 'Business Analyst']
    
    # Ensure minimum allocation for each role
        min_allocation = 0.05  # 5% minimum for each role
        team_composition = team_composition.numpy()[0]
        team_composition = np.maximum(team_composition, min_allocation)
        team_composition /= team_composition.sum()  # Renormalize
    
    # Adjust for software projects
        if project_requirements['project_type'].iloc[0] == 'Software':
            team_composition[roles.index('Developer')] *= 1.5  # Increase developer allocation
            team_composition /= team_composition.sum()  # Renormalize again
    
        return dict(zip(roles, team_composition))


    def train_pattern_recognizer(self, X, y, epochs=50, early_stop_patience=5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Convert pandas Series to numpy array if necessary
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        if isinstance(y_test, pd.Series):
            y_test = y_test.values
        
        X_train_float = np.array([self.scaler.fit_transform(x.astype(float)) for x in X_train])
        X_test_float = np.array([self.scaler.transform(x.astype(float)) for x in X_test])
        
        X_train = torch.FloatTensor(X_train_float)
        y_train = torch.FloatTensor(y_train.astype(float))
        X_test = torch.FloatTensor(X_test_float)
        y_test = torch.FloatTensor(y_test.astype(float))
        
        input_size = X_train.shape[2]
        hidden_size = 50
        output_size = 1
        self.pattern_recognizer = LSTMRegressor(input_size, hidden_size, output_size)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.pattern_recognizer.parameters(), lr=0.001)
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            self.pattern_recognizer.train()
            outputs = self.pattern_recognizer(X_train)
            loss = criterion(outputs.squeeze(), y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            self.pattern_recognizer.eval()
            with torch.no_grad():
                test_outputs = self.pattern_recognizer(X_test)
                test_loss = criterion(test_outputs.squeeze(), y_test)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")
            
            if test_loss < best_loss:
                best_loss = test_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print(f"Final Pattern Recognizer MSE: {best_loss:.4f}")

    def recognize_patterns(self, project_history):
        # Ensure project_history is a 3D array (batch_size, sequence_length, features)
        if len(project_history.shape) == 2:
            project_history = project_history.reshape(1, *project_history.shape)
        
        # Convert to float and apply the same scaling as in training
        project_history_float = np.array([self.scaler.transform(x.astype(float)) for x in project_history])
        
        with torch.no_grad():
            prediction = self.pattern_recognizer(torch.FloatTensor(project_history_float))
        return prediction.item()

   
   
    def simulate_project(self, initial_conditions, risk_score, success_probability, n_steps=12):
        print(f"Initial conditions: {initial_conditions}")  # Debug print
    
    # Normalize initial_conditions
        normalized_conditions = (initial_conditions - np.min(initial_conditions)) / (np.max(initial_conditions) - np.min(initial_conditions))
    
    # Use the mean of normalized conditions as the starting point, ensuring it's between 0 and 0.1
        current_progress = min(0.1, np.mean(normalized_conditions))
        print(f"Starting progress: {current_progress}")  # Debug print
    
        project_timeline = [current_progress]
        for step in range(n_steps - 1):
            print(f"Step {step + 1}")  # Debug print
        
        # Use the last 10 steps (or fewer if not available)
            last_10_steps = project_timeline[-10:]
        # Pad the input to match the expected sequence length (10)
            padded_input = np.pad(last_10_steps, (0, 10 - len(last_10_steps)), 'constant')
        # Reshape the input to (batch_size, sequence_length, input_size)
            input_tensor = torch.FloatTensor(padded_input).unsqueeze(0).unsqueeze(2).repeat(1, 1, 15)
        
            print(f"Input tensor shape: {input_tensor.shape}")  # Debug print
        
            with torch.no_grad():
                next_step = self.pattern_recognizer(input_tensor)
        
            next_progress = next_step.item()  # Get the scalar value
        
        # Add some random variability influenced by risk and success probability
            variability = np.random.normal(0, 0.05 * risk_score)
            next_progress += variability
        
        # Adjust progress based on success probability
            next_progress *= (0.5 + 0.5 * success_probability)
        
            print(f"Next progress: {next_progress}")  # Debug print
        
        # Ensure the progress is between current progress and 1, and generally increasing
            next_progress = max(current_progress, min(1, next_progress))
        
            project_timeline.append(next_progress)
            current_progress = next_progress
    
        return np.array(project_timeline)
    
    def summarize_data(self, project_data):
        # Create a copy of the data to avoid modifying the original
        data = project_data.copy()

        # Identify numeric and categorical columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        categorical_columns = data.select_dtypes(exclude=[np.number]).columns

    # One-hot encode categorical variables
        for col in categorical_columns:
             data = pd.get_dummies(data, columns=[col], prefix=col)

        # Calculate correlation matrix
        correlation_matrix = data.corr()

        # Generate summary statistics
        summary = data.describe()

    # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
        plt.title('Project Data Correlation Heatmap')
        plt.tight_layout()
        plt.show()

        return summary, correlation_matrix

    def get_team_image_paths(self, directory: str) -> List[str]:
        return [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    
    def analyze_team_sentiment(self, image_input):
        if isinstance(image_input, str):  # If it's a directory path
            image_paths = self.get_team_image_paths(image_input)
            images = [Image.open(path).convert("RGB") for path in image_paths]
        elif isinstance(image_input, UploadedFile):  # If it's an uploaded file
            images = [Image.open(io.BytesIO(image_input.getvalue())).convert("RGB")]
        else:
            raise ValueError("Invalid input type. Expected directory path or UploadedFile.")

        if not images:
            print(f"No images found in the input.")
            return None

        team_sentiments = []
        for image in images:
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)

            text_inputs = torch.cat([clip.tokenize(f"a photo of a {sentiment} team")
                                     for sentiment in ["happy", "neutral", "unhappy"]]).to(self.device)

            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_inputs)

                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            sentiments = ["happy", "neutral", "unhappy"]
            team_sentiments.append({sentiment: score.item() for sentiment, score in zip(sentiments, similarity[0])})

        # Calculate average sentiment across all images
        avg_sentiment = {
            "happy": np.mean([s["happy"] for s in team_sentiments]),
            "neutral": np.mean([s["neutral"] for s in team_sentiments]),
            "unhappy": np.mean([s["unhappy"] for s in team_sentiments])
        }

        return avg_sentiment
    
    def update_project_risk(self, project_features, team_sentiment):
        adjusted_features = project_features.copy()
        risk_factor = 1.0
        if team_sentiment["happy"] > 0.5:
            risk_factor = 0.9
        elif team_sentiment["unhappy"] > 0.5:
            risk_factor = 1.1
    
    # Adjust only specific features related to risk
        risk_related_features = ['complexity_High', 'budget', 'team_size']
        for feature in risk_related_features:
            if feature in adjusted_features.columns:
                adjusted_features[feature] *= risk_factor
    
        return self.classify_risk(adjusted_features)
    
# Usage example
pm_ai = ProjectManagementAI()
data = pm_ai.generate_synthetic_data(5000)

# Preprocess data
X = pm_ai.preprocess_data(data)
y_risk = data['risk_level']
y_effort = data['duration']
y_team = pd.get_dummies(data['team_size'])
y_pattern = data['success']

# Train models
pm_ai.train_risk_classifier(X, y_risk)
pm_ai.train_effort_estimator(X, y_effort)
pm_ai.train_team_generator(X.values, y_team.values)

# Prepare data for pattern recognition
pattern_data = np.array([X.values[i:i+10] for i in range(len(X)-10)])
pattern_target = y_pattern[10:]
pm_ai.train_pattern_recognizer(pattern_data, pattern_target)

# Example usage of trained models
new_project = pd.DataFrame({
    'duration': [180],
    'budget': [500000],
    'team_size': [10],
    'complexity': ['Medium'],
    'industry': ['Tech'],
    'project_type': ['Software'],
    'manager_experience': [5]
})
new_project_processed = pm_ai.preprocess_data(new_project, fit=False)

risk_probabilities = pm_ai.classify_risk(new_project_processed)
print("Risk probabilities:", risk_probabilities)

effort_estimate = pm_ai.estimate_effort(new_project_processed)
print("Estimated project duration:", effort_estimate)

team_composition = pm_ai.generate_team(new_project)  # Note: passing new_project, not new_project_processed
print("Recommended team composition:", team_composition)

project_history = pattern_data[-1:]  # Use the last project as history
success_probability = pm_ai.recognize_patterns(project_history)
print("Predicted success probability:", success_probability)

risk_score = 0.5  # Use an appropriate default value
success_probability = 0.5  # Use an appropriate default value
project_simulation = pm_ai.simulate_project(new_project_processed.values[0], risk_score, success_probability)
print("Project simulation:", project_simulation)

# Generate data summary
summary, correlation_matrix = pm_ai.summarize_data(data)
print("Summary Statistics:")
print(summary)
print("\nCorrelation Matrix:")
print(correlation_matrix)

 #Analyze team sentiment from images in a directory
team_image_directory = "./image_database"  # Replace with your actual image directory path
team_sentiment = pm_ai.analyze_team_sentiment(team_image_directory)
print("Team Sentiment Analysis:", team_sentiment)

# Update project risk based on team sentiment
updated_risk = pm_ai.update_project_risk(new_project_processed, team_sentiment)
print("Updated Risk Assessment:", updated_risk)
