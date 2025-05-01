from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd

# Load and preprocess
df = pd.read_csv('cleveland.csv', header=None)
df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol',
              'fbs', 'restecg', 'thalach', 'exang', 
              'oldpeak', 'slope', 'ca', 'thal', 'target']
df['target'] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
df['ca'] = df.ca.fillna(df.ca.mean())
df['thal'] = df.thal.fillna(df.thal.mean())
df['sex'] = df.sex.map({0: 'female', 1: 'male'})
df['sex'] = df['sex'].map({'female': 0, 'male': 1})
df = pd.get_dummies(df, columns=['thal'], drop_first=True)

X = df.drop('target', axis=1)
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_scaled = sc.transform(X)  # Full data for prediction

# Define base learners
svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
logreg = LogisticRegression(max_iter=1000)
nb = GaussianNB()
dt = DecisionTreeClassifier(max_depth=4)
rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', max_depth=3, learning_rate=0.1)
lgbm = LGBMClassifier(max_depth=3, learning_rate=0.1)

# Voting ensemble
ensemble = VotingClassifier(
    estimators=[
        ('svm', svm),
        ('logreg', logreg),
        ('nb', nb),
        ('rf', rf),
        ('xgb', xgb),
        ('lgbm', lgbm)
    ],
    voting='soft'
)

# Fit model
ensemble.fit(X_train, y_train)

# Predict
y_train_pred = ensemble.predict(X_train)
y_test_pred = ensemble.predict(X_test)

# Accuracy
train_acc = accuracy_score(y_train, y_train_pred) * 100
test_acc = accuracy_score(y_test, y_test_pred) * 100

print(f"\nImproved Ensemble Accuracy on Training Set: {train_acc:.2f}%")
print(f"Improved Ensemble Accuracy on Test Set: {test_acc:.2f}%")

# ------------------ NEW PART: Export only heart disease cases ------------------

# Predict on full dataset
predictions = ensemble.predict(X_scaled)

# Add predictions to original dataframe
df_with_preds = df.copy()
df_with_preds['predicted'] = predictions

# Filter only people predicted with heart disease (predicted = 1)
heart_disease_cases = df_with_preds[df_with_preds['predicted'] == 1]

# Export to CSV
heart_disease_cases.to_csv('heart_disease_cases.csv', index=False)
print("\nCSV file 'heart_disease_cases.csv' has been saved with predicted heart disease cases.")
