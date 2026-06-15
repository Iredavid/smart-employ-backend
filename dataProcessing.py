import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

dataset = pd.read_csv(
    BASE_DIR / "aaua_expanded_dataset.csv"
)

ALLOWED_FEATURES = [
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


def normalize(value, max_value):
    if max_value == 0:
        return 0
    return round(value / max_value, 3)


def build_career_requirements():

    requirements = {}

    careers = dataset["Career"].unique()

    for career in careers:

        career_data = dataset[
            dataset["Career"] == career
        ]

        # More realistic threshold
        threshold = career_data[
            "EmployabilityScore"
        ].quantile(0.65)

        top_students = career_data[
            career_data[
                "EmployabilityScore"
            ] >= threshold
        ]
        if len(top_students) < 10:
            top_students = career_data
        profile = {}

        raw_importance_scores = {}

        # =========================
        # FIRST PASS
        # =========================

        for feature in ALLOWED_FEATURES:

            feature_series = career_data[feature]
            score_series = career_data[
                "EmployabilityScore"
            ]

            top_freq = float(
                top_students[feature].mean()
            )

            overall_freq = float(
                career_data[feature].mean()
            )

            # =========================
            # IDEAL VALUES
            # =========================

            if feature == "CGPA":

                ideal = float(
                    top_students[feature].mean()
                )

            else:

                ideal = int(top_freq >= 0.6)

            minimum = float(
                top_students[
                    feature
                ].quantile(0.40)
            )

            # =========================
            # IMPORTANCE CALCULATION
            # =========================

            if feature_series.nunique() <= 1:

                importance = top_freq

            else:

                correlation = feature_series.corr(
                    score_series
                )

                correlation = (
                    abs(correlation)
                    if pd.notna(correlation)
                    else 0
                )

            
                competitive_advantage = max(
                    0,
                    top_freq - overall_freq
                ) 
                importance = (
                    (top_freq * 0.6)
                    +
                    (competitive_advantage * 0.3)
                    +
                    (correlation * 0.1)
                    # ((top_freq - overall_freq) * 0.2)
                )

            importance = max(0, importance)

            raw_importance_scores[
                feature
            ] = importance

            profile[feature] = {

                "ideal": round(
                    ideal,
                    3
                ),

                "minimum": round(
                    minimum,
                    3
                ),

                "frequency": round(
                    top_freq,
                    3
                ),

                "importance": importance
            }

        # =========================
        # NORMALIZATION
        # =========================

        max_importance = max(
            raw_importance_scores.values()
        )

        for feature in profile:

            profile[feature][
                "importance"
            ] = normalize(
                profile[feature]["importance"],
                max_importance
            )

        requirements[
            career
        ] = profile

    return requirements


requirements = (
    build_career_requirements()
)

joblib.dump(

    requirements,

    BASE_DIR /
    "career_requirements.pkl"
)

print(
    "career_requirements.pkl generated successfully"
)
