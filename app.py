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


FACULTY_SKILLS_RAW = {

    "Computing": [
        "Python",
        "Java",
        "Web Development",
        "Machine Learning",
        "Cybersecurity",
        "Database Management",
        "UI/UX Design",
        "Data Analysis",
        "Networking"
    ],

    "Science": [
        "Laboratory Skills",
        "Statistical Analysis",
        "Technical Writing",
        "Research Skills",
        "Critical Thinking"
    ],

    "Education": [
        "Teaching",
        "Counselling",
        "Classroom Management",
        "Curriculum Development",
        "Communication"
    ],

    "Social Sciences": [
        "Policy Analysis",
        "Argumentation",
        "Research Skills",
        "Critical Thinking",
        "Communication"
    ],

    "Management Sciences": [
        "Financial Analysis",
        "Project Management",
        "Marketing",
        "Accounting Software",
        "Communication"
    ],

    "Arts": [
        "Writing",
        "Content Creation",
        "Communication",
        "Critical Thinking",
        "Research Skills"
    ],

    "Agriculture": [
        "Field Work",
        "Environmental Analysis",
        "Research Skills",
        "Data Analysis"
    ],

    "Environmental Design": [
        "GIS",
        "Urban Planning",
        "Project Management",
        "Research Skills"
    ],

    "Law": [
        "Legal Research",
        "Legal Writing",
        "Argumentation",
        "Critical Thinking",
        "Communication"
    ]
}

FACULTY_SKILLS = {}

for faculty, skills in FACULTY_SKILLS_RAW.items():

    encoded_skills = []

    for skill in skills:

        encoded_feature = SKILL_MAP.get(skill)

        if encoded_feature:
            encoded_skills.append(
                encoded_feature
            )

    FACULTY_SKILLS[faculty] = list(
        set(encoded_skills)
    )


def generate_recommendations(
    input_df,
    career,
    department=None
):

    user = input_df.iloc[0].to_dict()

    target = career_requirements.get(
        career,
        {}
    )

    strengths = []
    improvements = []

    NON_BINARY_FEATURES = [
        "CGPA",
        "Internship"
    ]

    allowed_skills = []

    if department:
        allowed_skills = FACULTY_SKILLS.get(
            department,
            []
        )

    for feature, profile in target.items():

        current = user.get(feature, 0)

        minimum = profile.get("minimum", 0)

        ideal = profile.get("ideal", 1)

        importance = profile.get(
            "importance",
            0
        )

        frequency = profile.get(
            "frequency",
            0
        )

        # =========================
        # SKILL FEATURES
        # =========================

        if feature not in NON_BINARY_FEATURES:

            # Skip unrealistic skills ONLY if
            # importance is extremely low
            if (
                allowed_skills
                and feature not in allowed_skills
                and importance < 0.25
            ):
                continue

            # =========================
            # STRENGTHS
            # =========================

            if (
                current == 1
                and importance >= 0.2
            ):

                strengths.append({

                    "feature": feature,

                    "importance": importance,

                    "frequency": frequency,

                    "score": round(
                        importance * max(frequency, 0.3),
                        3
                    )
                })

            # =========================
            # IMPROVEMENTS
            # =========================

            elif current == 0:

                # Dynamic gap score
                gap_score = (
                    (
                        importance * 0.7
                    )
                    +
                    (
                        frequency * 0.3
                    )
                ) * 10

                # More forgiving threshold
                if gap_score >= 2:

                    improvements.append({

                        "feature": feature,

                        "gap": 1,

                        "priority": round(
                            gap_score,
                            2
                        ),

                        "target": "Recommended",

                        "importance": importance,

                        "frequency": frequency
                    })

        # =========================
        # CGPA
        # =========================

        elif feature == "CGPA":

            if current >= minimum:

                strengths.append({

                    "feature": feature,

                    "importance": importance,

                    "score": round(
                        current,
                        2
                    )
                })

            else:

                gap = max(
                    0,
                    ideal - current
                )

                priority = (
                    gap
                    * max(importance, 0.5)
                    * 12
                )

                improvements.append({

                    "feature": feature,

                    "gap": round(
                        gap,
                        2
                    ),

                    "priority": round(
                        priority,
                        2
                    ),

                    "target": round(
                        ideal,
                        2
                    ),

                    "importance": importance
                })

        # =========================
        # INTERNSHIP
        # =========================

        elif feature == "Internship":

            if current == 1:

                strengths.append({

                    "feature": feature,

                    "importance": importance,

                    "score": 1
                })

            else:

                improvements.append({

                    "feature": feature,

                    "gap": 1,

                    "priority": round(
                        max(
                            5,
                            importance * 10
                        ),
                        2
                    ),

                    "target": "Recommended",

                    "importance": importance
                })

    # =========================
    # FALLBACK SYSTEM
    # =========================

    # Guarantee recommendations always exist

    if len(improvements) == 0:

        fallback_features = sorted(

            target.items(),

            key=lambda x: (
                x[1].get(
                    "importance",
                    0
                )
            ),

            reverse=True
        )

        for feature, profile in fallback_features:

            if (
                feature not in NON_BINARY_FEATURES
                and user.get(feature, 0) == 0
            ):

                improvements.append({

                    "feature": feature,

                    "gap": 1,

                    "priority": 3,

                    "target": "Recommended",

                    "importance": profile.get(
                        "importance",
                        0
                    )
                })

            if len(improvements) >= 3:
                break

    # =========================
    # SORTING
    # =========================

    strengths.sort(

        key=lambda x: (
            x["importance"]
        ),

        reverse=True
    )

    improvements.sort(

        key=lambda x: (
            x["priority"]
        ),

        reverse=True
    )

    return {

        "strengths": strengths[:5],

        "gaps": improvements[:5]
    }

# def generate_recommendations(
#     input_df,
#     career,
#     department=None
# ):

#     user = input_df.iloc[0].to_dict()

#     target = career_requirements.get(
#         career,
#         {}
#     )

#     strengths = []
#     improvements = []

#     NON_BINARY_FEATURES = [
#         "CGPA",
#         "Internship"
#     ]

#     allowed_skills = []

#     if department:
#         allowed_skills = FACULTY_SKILLS.get(
#             department,
#             []
#         )

#     for feature, profile in target.items():

#         current = user.get(feature, 0)

#         minimum = profile.get("minimum", 0)

#         ideal = profile.get("ideal", 1)

#         importance = profile.get(
#             "importance",
#             0
#         )

#         frequency = profile.get(
#             "frequency",
#             0
#         )

#         # =========================
#         # SKILL FEATURES
#         # =========================

#         if feature not in NON_BINARY_FEATURES:

#             # suppress unrealistic skills
#             if (
#                 allowed_skills
#                 and feature not in allowed_skills
#                 and importance < 0.25
#             ):
#                 continue

#             # REAL strengths only
#             if (
#                 current == 1
#                 and importance >= 0.2
#                 # and frequency >= 0.5
#             ):

#                 strengths.append({

#                     "feature": feature,

#                     "importance": importance,

#                     "frequency": frequency,

#                     "score": round(
#                         importance * max(frequency, 0.3),
#                         3
#                     )
#                 })

#             # Missing important skills
#             elif (
#                 current == 0
#                 and importance >= 0.45
#                 and frequency >= 0.5
#             ):

#                 gap_score = (
#                     importance
#                     * frequency
#                     * 10
#                 )

#                 improvements.append({

#                     "feature": feature,

#                     "gap": 1,

#                     "priority": round(
#                         gap_score,
#                         2
#                     ),

#                     "target": "Recommended",

#                     "importance": importance,

#                     "frequency": frequency
#                 })

#         # =========================
#         # CGPA
#         # =========================

#         elif feature == "CGPA":

#             if current >= minimum:

#                 strengths.append({

#                     "feature": feature,

#                     "importance": importance,

#                     "score": round(
#                         current,
#                         2
#                     )
#                 })

#             else:

#                 gap = max(
#                     0,
#                     ideal - current
#                 )

#                 priority = (
#                     gap
#                     * importance
#                     * 15
#                 )

#                 improvements.append({

#                     "feature": feature,

#                     "gap": round(
#                         gap,
#                         2
#                     ),

#                     "priority": round(
#                         priority,
#                         2
#                     ),

#                     "target": round(
#                         ideal,
#                         2
#                     ),

#                     "importance": importance
#                 })

#         # =========================
#         # INTERNSHIP
#         # =========================

#         elif feature == "Internship":

#             if current == 1:

#                 strengths.append({

#                     "feature": feature,

#                     "importance": importance,

#                     "score": 1
#                 })

#             else:

#                 priority = max(
#                     6,
#                     importance * 10
#                 )

#                 improvements.append({

#                     "feature": feature,

#                     "gap": 1,

#                     "priority": round(
#                         priority,
#                         2
#                     ),

#                     "target": "Recommended",

#                     "importance": importance
#                 })

#     # =========================
#     # SORTING
#     # =========================

#     strengths.sort(

#         key=lambda x: (
#             x["importance"]
#         ),

#         reverse=True
#     )

#     improvements.sort(

#         key=lambda x: (
#             x["priority"]
#         ),

#         reverse=True
#     )

#     return {

#         "strengths": strengths[:5],

#         "gaps": improvements[:5]
#     }


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
            career,
            data.get("faculty")
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
