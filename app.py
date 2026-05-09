from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes
# Load models and metadata
clf = joblib.load(BASE_DIR / "career_model.pkl")
reg = joblib.load(BASE_DIR / "employability_model.pkl")
career_encoder = joblib.load(BASE_DIR / "career_encoder.pkl")
feature_columns = joblib.load(BASE_DIR / "feature_columns.pkl")
dataset = pd.read_csv(BASE_DIR / "aaua_production_dataset.csv")


def build_career_requirements():

    requirements = {}

    allowed_features = [
        "CGPA",
        "Internship",
        "Python",
        "Java",
        "WebDev",
        "DataAnalysis",
        "MachineLearning",
        "CyberSecurity",
        "Networking",
        "LabSkills",
        "Research",
        "Teaching",
        "ProjectManagement",
        "Communication",
        "Marketing",
        "GIS",
        "UIUX",
        "DatabaseManagement",
        "FinancialAnalysis",
        "CriticalThinking"
    ]

    careers = dataset["Career"].unique()

    for career in careers:

        career_data = dataset[
            dataset["Career"] == career
        ]

        # Top 20% performers
        threshold = career_data[
            "EmployabilityScore"
        ].quantile(0.80)

        top_students = career_data[
            career_data["EmployabilityScore"] >= threshold
        ]

        profile = {}

        for feature in allowed_features:

            profile[feature] = top_students[
                feature
            ].mean()

        requirements[career] = profile

    return requirements


career_requirements = build_career_requirements()


# 🔥 Function to prepare input correctly
def encode_skills(skills):
    features = {
        "Python": 0,
        "Java": 0,
        "WebDev": 0,
        "DataAnalysis": 0,
        "MachineLearning": 0,
        "CyberSecurity": 0,
        "Networking": 0,
        "LabSkills": 0,
        "Research": 0,
        "Teaching": 0,
        "ProjectManagement": 0,
        "Communication": 0,
        "Marketing": 0,
        "GIS": 0,
        "UIUX": 0,
        "DatabaseManagement": 0,
        "FinancialAnalysis": 0,
        "CriticalThinking": 0,
    }

    for skill in skills:
        s = skill.lower().strip()

        if "python" in s:
            features["Python"] = 1
        elif "java" in s:
            features["Java"] = 1
        elif "web" in s:
            features["WebDev"] = 1
        elif "data" in s or "excel" in s:
            features["DataAnalysis"] = 1
        elif "machine learning" in s:
            features["MachineLearning"] = 1
        elif "cyber" in s:
            features["CyberSecurity"] = 1
        elif "network" in s:
            features["Networking"] = 1
        elif "lab" in s:
            features["LabSkills"] = 1
        elif "research" in s:
            features["Research"] = 1
        elif "teach" in s:
            features["Teaching"] = 1
        elif "manage" in s:
            features["ProjectManagement"] = 1
        elif "communicat" in s or "speak" in s:
            features["Communication"] = 1
        elif "market" in s:
            features["Marketing"] = 1
        elif "gis" in s:
            features["GIS"] = 1
        elif "ux" in s:
            features["UI/UX Design"] = 1
        elif "database" in s:
            features["DatabaseManagement"] = 1
        elif "financial" in s:
            features["FinancialAnalysis"] = 1
        elif "critical" in s:
            features["CriticalThinking"] = 1

    return features


def prepare_input(data):
    # Extract data from React request
    cgpa = float(data.get("cgpa", 0))
    department = data.get("department", "")
    skills = data.get("skills", [])
    internship = 1 if data.get("internship") == "Yes" else 0

    # Encode skills
    skill_features = encode_skills(skills)

    # Create full feature dictionary
    input_dict = {col: 0 for col in feature_columns}

    # Fill core features
    input_dict["CGPA"] = cgpa
    input_dict["Internship"] = internship

    # Fill skill features
    for key, value in skill_features.items():
        if key in input_dict:
            input_dict[key] = value

    # 🔥 One-hot encode department
    dept_col = f"Department_{department}"
    if dept_col in input_dict:
        input_dict[dept_col] = 1
    else:
        return None  # invalid department

    # Convert to DataFrame (VERY IMPORTANT)
    return pd.DataFrame([input_dict])


def generate_recommendations(input_df, career):

    recommendations = []

    user = input_df.iloc[0].to_dict()

    target = career_requirements.get(career)

    if not target:
        return []

    for feature, ideal_value in target.items():

        user_value = user.get(feature, 0)

        if feature == "CGPA":

            if user_value < ideal_value:

                recommendations.append(
                    f"Improve CGPA (target: {round(ideal_value, 2)})"
                )

        elif feature == "Internship":

            if ideal_value >= 0.6 and user_value == 0:

                recommendations.append(
                    "Gain internship experience"
                )

        else:

            if ideal_value >= 0.6 and user_value == 0:

                recommendations.append(
                    f"Improve {feature}"
                )

    return recommendations[:5]


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        input_df = prepare_input(data)

        if input_df is None:
            return jsonify({"error": "Invalid department"}), 400

        # Predictions
        career_pred = clf.predict(input_df)[0]
        career = career_encoder.inverse_transform([career_pred])[0]

        # employability = reg.predict(input_df)[0]
        max_score = 110  # approximate max
        employability = int((reg.predict(input_df)[0] / max_score) * 100)
        recommendations = generate_recommendations(
            input_df,
            career
        )
        return jsonify({
            "career": career,
            "employability_score": round(float(employability), 2),
            "recommendations": recommendations
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
    app.run(debug=True)
