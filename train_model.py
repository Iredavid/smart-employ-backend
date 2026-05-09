import pandas as pd
import joblib

from xgboost import XGBClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV
)

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    r2_score
)


# =========================
# LOAD DATA
# =========================
data = pd.read_csv(
    "aaua_expanded_dataset.csv"
)


# =========================
# ENCODING
# =========================
le = LabelEncoder()

data["Career"] = le.fit_transform(
    data["Career"]
)

data = pd.get_dummies(
    data,
    columns=["Department"]
)


# =========================
# FEATURES
# =========================
X = data.drop(
    ["Career", "EmployabilityScore"],
    axis=1
)

y_class = data["Career"]
y_reg = data["EmployabilityScore"]


# =========================
# SPLIT
# =========================
(
    X_train,
    X_test,
    y_train_class,
    y_test_class,
    y_train_reg,
    y_test_reg
) = train_test_split(

    X,
    y_class,
    y_reg,

    test_size=0.2,

    stratify=y_class,

    random_state=42
)


print(
    "\nTraining shape:",
    X_train.shape
)


# =========================
# XGBOOST PIPELINE
# =========================
pipeline = Pipeline([

    (
        "smote",

        SMOTE(
            random_state=42,
            k_neighbors=1
        )
    ),

    (
        "model",

        XGBClassifier(

            objective="multi:softprob",

            eval_metric="mlogloss",

            num_class=len(
                le.classes_
            ),

            random_state=42,

            n_jobs=-1
        )
    )
])


# =========================
# HYPERPARAMETERS
# =========================
param_grid = {

    "model__n_estimators": [
        300,
        500,
        700
    ],

    "model__max_depth": [
        4,
        6,
        8
    ],

    "model__learning_rate": [
        0.03,
        0.05,
        0.1
    ],

    "model__subsample": [
        0.8,
        0.9,
        1.0
    ],

    "model__colsample_bytree": [
        0.8,
        0.9,
        1.0
    ],

    "model__min_child_weight": [
        1,
        3,
        5
    ],

    "model__gamma": [
        0,
        0.1,
        0.2
    ]
}


# =========================
# CROSS VALIDATION
# =========================
cv = StratifiedKFold(

    n_splits=3,

    shuffle=True,

    random_state=42
)


# =========================
# HYPERPARAMETER SEARCH
# =========================
search = RandomizedSearchCV(

    estimator=pipeline,

    param_distributions=param_grid,

    n_iter=10,

    scoring="accuracy",

    cv=cv,

    verbose=1,

    n_jobs=-1,

    random_state=42
)


search.fit(

    X_train,

    y_train_class
)


clf = search.best_estimator_


print(
    "\nBest Parameters:"
)

print(
    search.best_params_
)

print(
    "\nReal CV Accuracy:",
    search.best_score_
)


# =========================
# REGRESSOR
# =========================
reg = RandomForestRegressor(

    n_estimators=500,

    n_jobs=-1,

    random_state=42
)


reg.fit(

    X_train,

    y_train_reg
)


# =========================
# PREDICTIONS
# =========================
class_preds = clf.predict(
    X_test
)

reg_preds = reg.predict(
    X_test
)


# =========================
# FEATURE IMPORTANCE
# =========================
best_xgb = clf.named_steps[
    "model"
]


print(
    "\n===== TOP 15 FEATURES ====="
)

feature_importance = pd.DataFrame({

    "Feature": X.columns,

    "Importance":
    best_xgb.feature_importances_
})


feature_importance = (
    feature_importance
    .sort_values(
        by="Importance",
        ascending=False
    )
)

print(
    feature_importance.head(15)
)


# =========================
# METRICS
# =========================
print(
    "\n===== CLASSIFICATION REPORT ====="
)

print(

    classification_report(

        y_test_class,

        class_preds,

        target_names=le.classes_,

        zero_division=0
    )
)


print(
    "\n===== CONFUSION MATRIX ====="
)

print(

    confusion_matrix(

        y_test_class,

        class_preds
    )
)


print(
    "\n===== RESULTS ====="
)

print(
    "Cross-validation accuracy:",
    search.best_score_
)

print(
    "Test accuracy:",
    accuracy_score(
        y_test_class,
        class_preds
    )
)

print(
    "R2 score:",
    r2_score(
        y_test_reg,
        reg_preds
    )
)


# =========================
# SAVE
# =========================
joblib.dump(
    clf,
    "career_model.pkl"
)

joblib.dump(
    reg,
    "employability_model.pkl"
)

joblib.dump(
    le,
    "career_encoder.pkl"
)

joblib.dump(
    X.columns.tolist(),
    "feature_columns.pkl"
)


print(
    "\nModels saved successfully."
)

# =========================
# CLASSIFIER
# =========================
# clf = RandomForestClassifier(
#     n_estimators=500,
#     max_depth=None,
#     min_samples_split=2,
#     min_samples_leaf=1,
#     oob_score=True,
#     bootstrap=True,
#     n_jobs=-1,
#     random_state=42
# )

# clf.fit(
#     X_train_balanced,
#     y_train_balanced
# )


# import pandas as pd
# from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score
# import joblib

# # Load dataset
# data = pd.read_csv("aaua_production_dataset.csv")


# # Encode career labels
# le = LabelEncoder()
# data['Career'] = le.fit_transform(data['Career'])

# # Encode Department properly
# data = pd.get_dummies(data, columns=['Department'])

# # Features and targets
# X = data.drop(['Career', 'EmployabilityScore'], axis=1)
# y_class = data['Career']
# y_reg = data['EmployabilityScore']

# # Split data
# X_train, X_test, y_train_c, y_test_c = train_test_split(
#     X, y_class, test_size=0.2, random_state=42)
# _, _, y_train_r, y_test_r = train_test_split(
#     X, y_reg, test_size=0.2, random_state=42)

# # Train models
# clf = RandomForestClassifier(n_estimators=300,
#                              max_depth=20,
#                              class_weight="balanced",
#                              random_state=42,
#                              n_jobs=-1)
# clf.fit(X_train, y_train_c)

# reg = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
# reg.fit(X_train, y_train_r)


# # Evaluate
# # Cross-validation (fixed)
# cv_model = RandomForestClassifier(
#     n_estimators=200, max_depth=15, random_state=42)
# cv = StratifiedKFold(
#     n_splits=5,
#     shuffle=True,
#     random_state=42
# )
# cv_scores = cross_val_score(cv_model, X, y_class, cv=cv, scoring='accuracy')

# # Confusion Matrix
# cm = confusion_matrix(y_test_c, clf.predict(X_test))

# print(
#     classification_report(
#         y_test_c,
#         clf.predict(X_test),
#         target_names=le.classes_
#     )
# )

# print("Confusion Matrix:")
# print(cm)
# print("Classes:", le.classes_)

# print("Cross-validation accuracy:", cv_scores.mean())
# print("Test Accuracy:", accuracy_score(y_test_c, clf.predict(X_test)))
# print("R2 Score:", r2_score(y_test_r, reg.predict(X_test)))

# # Save everything
# joblib.dump(clf, "career_model.pkl")
# joblib.dump(reg, "employability_model.pkl")
# joblib.dump(le, "career_encoder.pkl")
# joblib.dump(X.columns.tolist(), "feature_columns.pkl")  # 🔥 VERY IMPORTANT
# print("Models trained and saved successfully!")
