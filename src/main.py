import pandas as pd
from sklearn.naive_bayes import GaussianNB

# Load data
data = pd.read_csv("data/symptoms.csv")

X = data.drop("disease", axis=1)
y = data["disease"]

# Train model
model = GaussianNB()
model.fit(X, y)

# User input
print("\nAnswer with 1 for YES and 0 for NO\n")

symptoms = {}
for col in X.columns:
    symptoms[col] = int(input(f"Do you have {col}? "))

input_data = pd.DataFrame([symptoms])

# Prediction
prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data).max()

print(f"\nPossible Disease: {prediction}")
print(f"Confidence: {round(probability * 100, 2)}%")

# Simple recommendation system
recommendations = {
    "Flu": "Rest and drink warm fluids.",
    "Cold": "Take vitamin C and rest.",
    "Malaria": "Seek medical attention immediately.",
    "Migraine": "Rest in dark room and avoid noise.",
    "COVID-19": "Isolate and get tested.",
    "Food_Poisoning": "Stay hydrated and avoid solid food."
}

print("Recommendation:", recommendations[prediction])

