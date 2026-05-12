import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

dataset = pd.read_csv(
    BASE_DIR / "aaua_production_dataset.csv"
)


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

        threshold = career_data[
            "EmployabilityScore"
        ].quantile(0.80)

        top_students = career_data[
            career_data["EmployabilityScore"] >= threshold
        ]

        profile = {}

        for feature in allowed_features:

            profile[feature] = (
                top_students[feature].mean()
            )

        requirements[career] = profile

    return requirements


requirements = build_career_requirements()

joblib.dump(
    requirements,
    BASE_DIR / "career_requirements.pkl"
)

print("Done!")