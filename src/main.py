import pandas as pd
from sklearn.naive_bayes import GaussianNB

# Load data
data = pd.read_csv("data/symptoms.csv")

X = data.drop("disease", axis=1)
y = data["disease"]

# Train model
model = GaussianNB()
model.fit(X, y)

print("\nAnswer with yes/no or 1/0\n")

# User input
symptoms = {}

for col in X.columns:
    while True:
        ans = input(f"Do you have {col}? ").strip().lower()

        if ans in ["yes", "y", "1"]:
            symptoms[col] = 1
            break
        elif ans in ["no", "n", "0"]:
            symptoms[col] = 0
            break
        else:
            print("Please answer with yes, no, 1 or 0.")

input_data = pd.DataFrame([symptoms])

# Prediction
probs = model.predict_proba(input_data)[0]
diseases = model.classes_

results = list(zip(diseases, probs))
results.sort(key=lambda x: x[1], reverse=True)

print("\nTop Possible Diseases:\n")
for disease, prob in results[:3]:
    print(f"{disease}: {round(prob * 100, 2)}%")

prediction = results[0][0]

# Recommendation system
recommendations = {
    "Flu": "Rest and drink warm fluids.",
    "Cold": "Take vitamin C and rest.",
    "Malaria": "Seek medical attention immediately.",
    "Migraine": "Rest in a dark room and avoid noise.",
    "COVID-19": "Isolate and get tested.",
    "Food_Poisoning": "Stay hydrated and avoid solid food."
}

print("\nRecommendation:")
print(recommendations[prediction])

