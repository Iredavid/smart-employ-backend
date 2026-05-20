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
career_requirements = joblib.load(BASE_DIR / "career_requirements.pkl")

# 🔥 Function to prepare input correctly
FEATURE_LABELS = {
    "CGPA": "Academic Performance",
    "Internship": "Industry Experience",

    "Python": "Python Programming",
    "Java": "Java Development",
    "WebDev": "Web Development",
    "DataAnalysis": "Data Analysis",
    "MachineLearning": "Machine Learning",
    "CyberSecurity": "Cybersecurity",
    "Networking": "Networking",

    "LabSkills": "Laboratory Skills",
    "Research": "Research Skills",
    "Teaching": "Teaching Skills",

    "ProjectManagement": "Project Management",
    "Communication": "Communication Skills",

    "Marketing": "Marketing",
    "GIS": "GIS Analysis",

    "UIUX": "UI/UX Design",

    "DatabaseManagement": "Database Management",

    "FinancialAnalysis": "Financial Analysis",

    "CriticalThinking": "Critical Thinking"
}

SKILL_MAP = {

    # Universal skills
    "Communication": "Communication",
    "Research Skills": "Research",
    "Critical Thinking": "CriticalThinking",
    "Data Analysis": "DataAnalysis",
    "Networking": "Networking",

    # Computing
    "Python": "Python",
    "Java": "Java",
    "Web Development": "WebDev",
    "Machine Learning": "MachineLearning",
    "Cybersecurity": "CyberSecurity",
    "Database Management": "DatabaseManagement",
    "UI/UX Design": "UIUX",

    # Science
    "Laboratory Skills": "LabSkills",
    "Statistical Analysis": "DataAnalysis",
    "Technical Writing": "Research",

    # Education
    "Teaching": "Teaching",
    "Counselling": "Communication",
    "Classroom Management": "ProjectManagement",
    "Curriculum Development": "Teaching",

    # Social Sciences
    "Policy Analysis": "CriticalThinking",

    # Management
    "Financial Analysis": "FinancialAnalysis",
    "Project Management": "ProjectManagement",
    "Marketing": "Marketing",
    "Accounting Software": "FinancialAnalysis",

    # Arts
    "Writing": "Communication",
    "Content Creation": "Communication",

    # Agriculture
    "Field Work": "LabSkills",
    "Environmental Analysis": "Research",

    # Environmental Design
    "GIS": "GIS",
    "Urban Planning": "ProjectManagement",

    # Law
    "Legal Research": "Research",
    "Legal Writing": "Communication",
    "Argumentation": "CriticalThinking",
}


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
        model_feature = SKILL_MAP.get(
            skill
        )

        if model_feature:

            features[
                model_feature
            ] = 1

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

    user = input_df.iloc[0].to_dict()

    target = career_requirements.get(career, {})

    strengths = []
    improvements = []

    NON_BINARY_FEATURES = [
        "CGPA",
        "Internship"
    ]

    for feature, profile in target.items():

        current = user.get(feature, 0)

        minimum = profile.get("minimum", 0)
        ideal = profile.get("ideal", 1)
        importance = profile.get("importance", 1)

        # =========================
        # SKILL FEATURES
        # =========================
        if feature not in NON_BINARY_FEATURES:

            # User has skill
            if current == 1:

                strengths.append({
                    "feature": feature,
                    "score": current,
                    "importance": importance
                })

            # User lacks skill
            else:

                # ONLY recommend important skills
                if importance >= 0.6:

                    improvements.append({
                        "feature": feature,
                        "gap": 1,
                        "priority": round(importance * 10, 2),
                        "target": "Required",
                        "importance": importance
                    })

        # =========================
        # CGPA
        # =========================
        elif feature == "CGPA":

            if current >= minimum:

                strengths.append({
                    "feature": feature,
                    "score": round(current, 2),
                    "importance": importance
                })

            else:

                gap = max(0, ideal - current)

                # HIGHER PRIORITY WEIGHT
                priority = gap * importance * 15

                improvements.append({
                    "feature": feature,
                    "gap": round(gap, 2),
                    "priority": round(priority, 2),
                    "target": round(ideal, 2),
                    "importance": importance
                })

        # =========================
        # INTERNSHIP
        # =========================
        elif feature == "Internship":

            # User HAS internship
            if current == 1:

                strengths.append({
                    "feature": feature,
                    "score": current,
                    "importance": importance
                })

            # User DOES NOT have internship
            else:

                # ONLY recommend if internship is important
                if importance >= 0.7:

                    improvements.append({
                        "feature": feature,
                        "gap": 1,
                        "priority": round(importance * 8, 2),
                        "target": "Recommended",
                        "importance": importance
                    })

    # =========================
    # SORTING
    # =========================

    strengths.sort(
        key=lambda x: x["importance"],
        reverse=True
    )

    improvements.sort(
        key=lambda x: x["priority"],
        reverse=True
    )

    return {
        "strengths": strengths[:5],
        "gaps": improvements[:5]
    }


def generate_feedback(career, employability, analysis):

    strengths = [
        FEATURE_LABELS.get(s["feature"], s["feature"])
        for s in analysis["strengths"][:4]
    ]

    gaps = analysis["gaps"]

    actions = []

    for gap in gaps:
        skill = FEATURE_LABELS.get(gap["feature"], gap["feature"])

        feature = gap["feature"]

        priority = "Medium"

        if gap["priority"] >= 10:
            priority = "High"

        elif gap["priority"] <= 5:
            priority = "Low"

       # =========================
       # CGPA
       # =========================
        if feature == "CGPA":

            actions.append({
                "title": "Improve Academic Performance",
                "detail": (
                    f"Focus on improving your CGPA toward "
                    f"{gap['target']} to strengthen eligibility "
                    f"for competitive opportunities."
                ),
                "priority": priority
            })

    # =========================
    # INTERNSHIP
    # =========================
        elif feature == "Internship":

            actions.append({
                "title": "Gain Industry Experience",
                "detail": (
                    "Participate in internships, SIWES, "
                    "or real-world projects to improve "
                    "industry readiness."
                ),
                "priority": priority
            })

    # =========================
    # SKILLS
    # =========================
        else:

            actions.append({
                "title": f"Develop {skill}",
                "detail": (
                    f"Build stronger {skill.lower()} capability "
                    "through hands-on projects, certifications, "
                    "and practical experience."
                ),
                "priority": priority
            })

    return {
        "summary": {
            "career": career,
            "employability_score": f"{employability}%",
            "message": (
                f"You show a strong alignment with a {career} pathway. "
                f"Your profile demonstrates solid potential with clear growth areas."
            )
        },

        "strengths": strengths,

        "action_plan": actions,

        "closing_note": (
            "Improving the highlighted areas will significantly increase your competitiveness "
            f"for a {career} role."
        )
    }


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

        max_score = 110  # approximate max
        employability = int((reg.predict(input_df)[0] / max_score) * 100)
        recommendations = generate_recommendations(
            input_df,
            career
        )

        return jsonify({
            "career": career,
            "employability_score": round(float(employability), 2),
            "recommendations": generate_feedback(
                career,
                employability,
                recommendations
            )
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
