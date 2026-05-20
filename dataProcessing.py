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


def build_career_requirements():

    requirements = {}

    careers = dataset[
        "Career"
    ].unique()

    for career in careers:

        career_data = dataset[
            dataset["Career"] == career
        ]

        threshold = career_data[
            "EmployabilityScore"
        ].quantile(0.80)

        top_students = career_data[
            career_data[
                "EmployabilityScore"
            ] >= threshold
        ]

        profile = {}

        for feature in ALLOWED_FEATURES:

            if feature in ["CGPA"]:
                ideal = float(
                    top_students[feature].mean()
                )

            else:
                ideal = int(
                    top_students[feature].mean() >= 0.6
                )

            minimum = float(
                top_students[
                    feature
                ].quantile(0.40)
            )

            frequency = float(
                top_students[
                    feature
                ].mean()
            )
            feature_series = career_data[
                feature
            ]

            score_series = career_data[
                "EmployabilityScore"
            ]

            if feature_series.nunique() <= 1:
                importance = 1.0
            else:
                correlation = feature_series.corr(
                    score_series
                )
                importance = float(
                    abs(correlation)
                ) if pd.notna(
                    correlation
                ) else 0.0

            profile[
                feature
            ] = {

                "ideal": round(
                    ideal,
                    3
                ),

                "minimum": round(
                    minimum,
                    3
                ),

                "frequency": round(
                    frequency,
                    3
                ),

                "importance": round(
                    importance,
                    3
                )
            }

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
