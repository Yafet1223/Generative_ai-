import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv("luma_dataset.csv")
df = df.drop(columns=["date"])

FEATURES = [
    "is_church_day",
    "wake_up_time_hr",
    "sleep_hours",
    "sleep_quality_1to10",
    "morning_prayer_done",
    "breakfast_eaten",
    "exercise_done",
    "exercise_minutes",
    "classes_attended",
    "stress_level_1to10",
    "mood_morning_1to10",
    "caffeine_intake_cups",
    "tasks_planned",
    "day_of_week"
]

TARGET = "task_completion_likelihood"

X = df[FEATURES]
y = df[TARGET]

# ─────────────────────────────────────────────
# 2. ENCODE CATEGORICAL
# ─────────────────────────────────────────────
X = pd.get_dummies(X, columns=["day_of_week"])
TRAINING_COLUMNS = X.columns.tolist()

# ─────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────────
# 4. MODEL PIPELINE
# ─────────────────────────────────────────────
model = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    ))
])

# ─────────────────────────────────────────────
# 5. TRAIN
# ─────────────────────────────────────────────
model.fit(X_train, y_train)

# ─────────────────────────────────────────────
# 6. EVALUATE
# ─────────────────────────────────────────────
score = model.score(X_test, y_test)
print("\nTest R²:", round(score, 4))

# ─────────────────────────────────────────────
# 7. SAVE MODEL
# ─────────────────────────────────────────────
joblib.dump(model, "luma_model.pkl")
joblib.dump(TRAINING_COLUMNS, "luma_columns.pkl")

print("Model saved.")

# ─────────────────────────────────────────────
# 8. PREDICTION FUNCTION
# ─────────────────────────────────────────────
def predict_day(input_dict):
    model = joblib.load("luma_model.pkl")
    columns = joblib.load("luma_columns.pkl")

    df = pd.DataFrame([input_dict])

    # encode
    df = pd.get_dummies(df, columns=["day_of_week"])

    # align to training columns
    df = df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(df)[0]

    return round(float(prediction), 2)

# ─────────────────────────────────────────────
# 9. TEST EXAMPLE
# ─────────────────────────────────────────────
today = {
    "is_church_day": 0,
    "wake_up_time_hr": 6.5,
    "sleep_hours": 7.5,
    "sleep_quality_1to10": 8,
    "morning_prayer_done": 1,
    "breakfast_eaten": 1,
    "exercise_done": 1,
    "exercise_minutes": 30,
    "classes_attended": 4,
    "stress_level_1to10": 3,
    "mood_morning_1to10": 8,
    "caffeine_intake_cups": 1,
    "tasks_planned": 6,
    "day_of_week": "Wednesday"
}

print("\nPrediction:", predict_day(today))