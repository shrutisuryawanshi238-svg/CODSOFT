import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("Titanic-Dataset.csv")

# Convert text to numbers
le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])

# Fill missing Age values
data['Age'] = data['Age'].fillna(data['Age'].median())

# Features and Target
X = data[['Pclass', 'Sex', 'Age', 'Fare']]
y = data['Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Titanic Survival Prediction Accuracy:", accuracy)