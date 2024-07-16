from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load or train encoders
sex_label_encoder = LabelEncoder()
sex_label_encoder.fit(['male', 'female'])

embarked_label_encoder = LabelEncoder()
embarked_label_encoder.fit(['S', 'C', 'Q'])

# Define models
models = {
    'svc': ('model_svc.pkl', SVC(gamma='auto', random_state=0)),
    'logistic_regression': ('model_logistic_regression.pkl', LogisticRegression()),
    'decision_tree': ('model_decision_tree.pkl', DecisionTreeClassifier())
}

# Train models if they don't exist
def train_and_save_model(model_name, model, train_x, train_y):
    print(f"Training {model_name}...")
    model.fit(train_x, train_y)
    with open(models[model_name][0], 'wb') as f:
        pickle.dump(model, f)
    print(f"{model_name} model saved as {models[model_name][0]}")

# Check if models exist, otherwise train and save them
for model_name, (model_file, model) in models.items():
    if not os.path.exists(model_file):
        # Load dataset for training
        sample_data = pd.read_csv('static/dataset/train.csv')
        
        # Preprocess dataset
        sample_data['Age'] = sample_data['Age'].fillna(sample_data['Age'].median()).astype('int')
        sample_data['Embarked'] = sample_data['Embarked'].fillna(sample_data['Embarked'].mode()[0])
        sample_data = sample_data.drop(columns=['Cabin', 'PassengerId', 'Ticket', 'Fare', 'Name'], axis=1)
        sample_data['Sex'] = sex_label_encoder.transform(sample_data['Sex'])
        sample_data['Embarked'] = embarked_label_encoder.transform(sample_data['Embarked'])
        
        # Split data into train and test
        train_x, _, train_y, _ = train_test_split(sample_data.iloc[:, 1:], sample_data.iloc[:, 0], test_size=0.2, random_state=0)
        
        # Train and save model
        train_and_save_model(model_name, model, train_x, train_y)
    else:
        with open(model_file, 'rb') as f:
            models[model_name] = pickle.load(f)

@app.route('/')
def index():
    generate_classifier_comparison_charts()
    return render_template('index.html', result=None)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    pclass = int(data['pclass'])
    sex = data['sex']
    age = int(data['age'])
    sibsp = int(data['sibsp'])
    parch = int(data['parch'])
    embarked = data['embarked']
    classifier = data['classifier']

    # Transform input
    sex_encoded = sex_label_encoder.transform([sex])[0]
    embarked_encoded = embarked_label_encoder.transform([embarked])[0]

    # Create input array
    input_features = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex_encoded],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Embarked': [embarked_encoded]
    })

    # Predict
    model = models[classifier]
    prediction = model.predict(input_features)[0]
    result = 'Survived' if prediction == 1 else 'Not Survived'

    return jsonify({'result': result})

def generate_classifier_comparison_charts():
    # Load dataset for scoring
    sample_data = pd.read_csv('static/dataset/train.csv')
    
    # Preprocess dataset
    sample_data['Age'] = sample_data['Age'].fillna(sample_data['Age'].median()).astype('int')
    sample_data['Embarked'] = sample_data['Embarked'].fillna(sample_data['Embarked'].mode()[0])
    sample_data = sample_data.drop(columns=['Cabin', 'PassengerId', 'Ticket', 'Fare', 'Name'], axis=1)
    sample_data['Sex'] = sex_label_encoder.transform(sample_data['Sex'])
    sample_data['Embarked'] = embarked_label_encoder.transform(sample_data['Embarked'])
    
    # Split data into train and test
    train_x, _, train_y, _ = train_test_split(sample_data.iloc[:, 1:], sample_data.iloc[:, 0], test_size=0.2, random_state=0)

    # Get classifier scores
    scores = {}
    for model_name, model in models.items():
        score = cross_val_score(model, train_x, train_y, cv=5, scoring='accuracy').mean()
        scores[model_name] = score

    # Generate bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(scores.keys(), scores.values(), color=['red', 'blue', 'green'])
    plt.xlabel('Classifier')
    plt.ylabel('Accuracy')
    plt.title('Classifier Accuracy Comparison')

    # Annotate bars with accuracy values
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 3), ha='center', va='bottom')

    plt.savefig('static/classifier_accuracy_comparison.png')
    plt.close()

    # Generate pie chart
    plt.figure(figsize=(10, 6))
    plt.pie(scores.values(), labels=scores.keys(), autopct='%1.1f%%', colors=['red', 'blue', 'green'])
    plt.title('Classifier Accuracy Distribution')
    plt.savefig('static/classifier_accuracy_distribution.png')
    plt.close()

    # Generate line chart
    plt.figure(figsize=(10, 6))
    plt.plot(scores.keys(), scores.values(), marker='o', linestyle='-', color='blue')
    plt.xlabel('Classifier')
    plt.ylabel('Accuracy')
    plt.title('Classifier Accuracy Trend')
    plt.savefig('static/classifier_accuracy_trend.png')
    plt.close()

if __name__ == '__main__':
    # app.run(debug=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=5000)
