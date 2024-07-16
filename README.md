# Titanic Passenger Survival Prediction
Predicting survival of Titanic passengers using machine learning classifiers.

## Features
  - Prediction Form: Allows users to input passenger details and select a classifier for survival prediction.
  - Classifier Comparison: Displays comparison graphs for different classifiers based on accuracy metrics.
  - Prediction Result: Provides the predicted survival outcome for the input passenger details.


## Installation
1. Clone the repository
```
    git clone https://github.com/RasikhAli/Titanic-Passenger-Survival
    cd Titanic-Passenger-Survival
```
2. Install dependencies:
```
  pip install -r requirements.txt
```
3. Run the application:
```
  python app.py
```
The application will run on http://localhost:5000 by default.


## Usage
1. Fill Prediction Form:
     - Select ticket class, gender, age, number of siblings/spouses, number of parents/children, port of embarkation, and classifier.
     - Click on "Predict" to get the survival prediction.

2. Compare Classifiers:
     - Select different types of graphs (accuracy comparison, distribution, trend) to compare classifier performance.

3. View Prediction Result:
     - The predicted survival outcome for the input passenger details will be displayed.


## Folder Structure
```
    ├── static/
    │   ├── classifier_accuracy_comparison.png      # Example graph image
    │   ├── classifier_accuracy_distribution.png   # Example graph image
    │   ├── classifier_accuracy_trend.png          # Example graph image
    │   └── ...                                    # Other static files
    ├── templates/
    │   ├── index.html                             # Main HTML template
    │   └── ...                                    # Other templates
    ├── app.py                                     # Flask application
    ├── model_svc.pkl                              # SVC model
    ├── model_lr.pkl                               # LR  model
    ├── model_dt.pkl                               # DT  model
    ├── requirements.txt                           # Python dependencies
    └── README.md                                  # Project documentation
```

## Technologies Used
  - Flask: Python web framework for building the application.
  - scikit-learn: Machine learning library for implementing classifiers.
  - matplotlib: Library for plotting classifier comparison graphs.
  - HTML/CSS/Bootstrap: Frontend technologies for the user interface.
