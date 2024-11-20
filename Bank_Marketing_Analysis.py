# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load and preprocess the data
def load_data(filepath):
    
    data = pd.read_csv(filepath, sep=';')
    # One-Hot Encoding
    categorical_cols = data.select_dtypes(include=['object']).columns
    data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    return data_encoded

# Apply scaling to numerical columns
def scale_data(data, numerical_columns):
    
    scaler = StandardScaler()
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    return data

# Perform feature selection using SelectKBest
def select_features(X, y, k=10):
    
    selector = SelectKBest(score_func=f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    return X_new, selected_features

# Train and evaluate models
def train_models(X_train, y_train, X_test, y_test):
    
    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)  # Train
    y_pred_log = log_reg.predict(X_test)  # Predict
    metrics_log = evaluate_model(y_test, y_pred_log, "Logistic Regression")

    # Decision Tree
    dec_tree = DecisionTreeClassifier(random_state=42)
    dec_tree.fit(X_train, y_train)  # Train
    y_pred_tree = dec_tree.predict(X_test)  # Predict
    metrics_tree = evaluate_model(y_test, y_pred_tree, "Decision Tree")

    # Random Forest
    random_forest = RandomForestClassifier(random_state=42, n_estimators=100)
    random_forest.fit(X_train, y_train)  # Train
    y_pred_rf = random_forest.predict(X_test)  # Predict
    metrics_rf = evaluate_model(y_test, y_pred_rf, "Random Forest")

    return {
        "Logistic Regression": {"model": log_reg, "metrics": metrics_log},
        "Decision Tree": {"model": dec_tree, "metrics": metrics_tree},
        "Random Forest": {"model": random_forest, "metrics": metrics_rf}
    }

# Evaluate a model
def evaluate_model(y_true, y_pred, model_name):
   
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# Perform ensemble prediction
def ensemble_prediction(models, X_train, y_train, X_test, y_test):
   
    ensemble = VotingClassifier(estimators=[
        ('lr', models['Logistic Regression']['model']),
        ('dt', models['Decision Tree']['model']),
        ('rf', models['Random Forest']['model'])
    ], voting='soft')  # Soft voting
    
    ensemble.fit(X_train, y_train)  # Train ensemble
    y_pred_ensemble = ensemble.predict(X_test)  # Predict
    accuracy = accuracy_score(y_test, y_pred_ensemble)
    print(f"\nEnsemble Accuracy: {accuracy:.4f}")
    return y_pred_ensemble

# Plot prediction results
def plot_predictions(y_pred):
    
    predictions_count = pd.Series(y_pred).value_counts()
    predictions_count.plot(kind='bar', color=['red', 'green'])
    plt.title("Predicted Term Deposit Subscriptions")
    plt.xlabel("Subscription Outcome")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=0)
    plt.show()
def cross_validate_model(model, X, y):
    
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')  # 5-fold CV
    print(f"\n{model.__class__.__name__} Cross-Validation Results:")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
    print(f"CV Accuracy Standard Deviation: {cv_scores.std():.4f}")
    return cv_scores.mean(), cv_scores.std()

# Main script execution
if __name__ == "__main__":
    # Load and preprocess data
    filepath = 'your filepath'
    bank_data = load_data(filepath)
    numerical_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
    bank_data = scale_data(bank_data, numerical_columns)
    
    # Split data into features and target
    X = bank_data.drop(columns=['y_yes'])
    y = bank_data['y_yes']
    
    # Select features
    X_new, selected_features = select_features(X, y, k=10)
    print(f"Selected Features: {list(selected_features)}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)
    
    # Train models
    models_results = train_models(X_train, y_train, X_test, y_test)
    
    # Perform cross-validation for each model
    for model_name, model_data in models_results.items():
        model = model_data["model"]
        cross_validate_model(model, X_new, y)  # Perform 5-fold cross-validation
    
    # Ensemble prediction
    y_pred_ensemble = ensemble_prediction(models_results, X_train, y_train, X_test, y_test)
    
    # Plot final predictions
    plot_predictions(y_pred_ensemble)
    



