import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score

# Load data
df = pd.read_csv("luma_dataset.csv")
df = df.drop(columns=["date"])

TARGET = "task_completion_likelihood"
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

# Load model and columns
model = joblib.load("luma_model.pkl")
columns = joblib.load("luma_columns.pkl")

def predict_day(input_dict):
    df_input = pd.DataFrame([input_dict])
    df_input = pd.get_dummies(df_input, columns=["day_of_week"])
    df_input = df_input.reindex(columns=columns, fill_value=0)
    prediction = model.predict(df_input)[0]
    return round(float(prediction), 2)

# Prepare data for plots
X = df[FEATURES]
y = df[TARGET]
X_encoded = pd.get_dummies(X, columns=["day_of_week"])
X_encoded = X_encoded.reindex(columns=columns, fill_value=0)

# Predictions for residuals
y_pred = model.predict(X_encoded)

st.title("Task Completion Likelihood Regression Dashboard")

# Tabs for different sections
tab1, tab2, tab3 = st.tabs(["Data Insights", "Daily Prediction", "Weekly Planner"])

with tab1:
    st.header("Data Overview")
    st.write("Sample of the dataset:")
    st.dataframe(df.head())

    st.header("Correlation Heatmap")
    corr = df.select_dtypes(include=['number']).corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto")
    st.plotly_chart(fig_corr)

    st.header("Scatter Plots: Features vs Target")
    selected_feature = st.selectbox("Select a feature to plot against target:", FEATURES[:-1])  # exclude day_of_week for simplicity
    fig_scatter = px.scatter(df, x=selected_feature, y=TARGET, trendline="ols")
    st.plotly_chart(fig_scatter)

    st.header("Model Performance")
    r2 = r2_score(y, y_pred)
    st.write(f"R² Score: {r2:.4f}")

    st.header("Residuals Plot")
    residuals = y - y_pred
    fig_resid = px.scatter(x=y_pred, y=residuals, labels={'x': 'Predicted', 'y': 'Residuals'})
    fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_resid)

    st.header("Feature Importance")
    importance = model.named_steps['rf'].feature_importances_
    feat_imp = pd.DataFrame({'Feature': columns, 'Importance': importance}).sort_values('Importance', ascending=False)
    fig_imp = px.bar(feat_imp, x='Importance', y='Feature', orientation='h')
    st.plotly_chart(fig_imp)

with tab2:
    st.header("Predict Your Task Completion Likelihood")
    st.write("Answer the questions below about your daily schedule and habits to get a prediction.")

    # Use session state to store inputs
    if 'prediction' not in st.session_state:
        st.session_state.prediction = None

    with st.form("schedule_form"):
        st.subheader("Daily Schedule Questions")
        is_church_day = st.checkbox("Is today a church day?")
        wake_up_time_hr = st.slider("What time did you wake up? (hours, e.g., 6.5 for 6:30 AM)", 4.0, 12.0, 7.0)
        sleep_hours = st.slider("How many hours did you sleep last night?", 4.0, 12.0, 8.0)
        sleep_quality = st.slider("Rate your sleep quality (1-10)", 1, 10, 7)
        morning_prayer = st.checkbox("Did you do morning prayer?")
        breakfast = st.checkbox("Did you eat breakfast?")
        exercise = st.checkbox("Did you do exercise today?")
        exercise_minutes = st.slider("How many minutes of exercise?", 0, 120, 30)
        classes_attended = st.slider("How many classes did you attend today?", 0, 10, 5)
        stress_level = st.slider("Rate your stress level (1-10)", 1, 10, 5)
        mood_morning = st.slider("Rate your morning mood (1-10)", 1, 10, 7)
        caffeine = st.slider("How many cups of caffeine did you have?", 0, 5, 1)
        tasks_planned = st.slider("How many tasks did you plan for today?", 0, 20, 5)
        day_of_week = st.selectbox("What day of the week is it?", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

        submitted = st.form_submit_button("Get Prediction")
        if submitted:
            input_dict = {
                "is_church_day": int(is_church_day),
                "wake_up_time_hr": wake_up_time_hr,
                "sleep_hours": sleep_hours,
                "sleep_quality_1to10": sleep_quality,
                "morning_prayer_done": int(morning_prayer),
                "breakfast_eaten": int(breakfast),
                "exercise_done": int(exercise),
                "exercise_minutes": exercise_minutes,
                "classes_attended": classes_attended,
                "stress_level_1to10": stress_level,
                "mood_morning_1to10": mood_morning,
                "caffeine_intake_cups": caffeine,
                "tasks_planned": tasks_planned,
                "day_of_week": day_of_week
            }
            st.session_state.prediction = predict_day(input_dict)

    if st.session_state.prediction is not None:
        st.success(f"Your predicted Task Completion Likelihood: {st.session_state.prediction}")
        st.write("This score is based on your inputs and the trained model. Higher scores indicate better likelihood of completing tasks.")

with tab3:
    st.header("Weekly Planner (Monday to Friday)")
    st.write("Plan your week by entering schedules for each weekday and get predictions for task completion likelihood.")

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    predictions = {}

    with st.form("weekly_form"):
        for day in days:
            st.subheader(f"{day} Schedule")
            col1, col2 = st.columns(2)
            with col1:
                wake_up = st.slider(f"Wake up time ({day})", 4.0, 12.0, 7.0, key=f"wake_{day}")
                sleep_hrs = st.slider(f"Sleep hours ({day})", 4.0, 12.0, 8.0, key=f"sleep_{day}")
                sleep_qual = st.slider(f"Sleep quality ({day})", 1, 10, 7, key=f"qual_{day}")
                prayer = st.checkbox(f"Morning prayer ({day})", key=f"prayer_{day}")
                breakfast = st.checkbox(f"Breakfast ({day})", key=f"breakfast_{day}")
            with col2:
                exercise = st.checkbox(f"Exercise ({day})", key=f"exercise_{day}")
                ex_min = st.slider(f"Exercise minutes ({day})", 0, 120, 30, key=f"exmin_{day}")
                classes = st.slider(f"Classes attended ({day})", 0, 10, 5, key=f"classes_{day}")
                stress = st.slider(f"Stress level ({day})", 1, 10, 5, key=f"stress_{day}")
                mood = st.slider(f"Morning mood ({day})", 1, 10, 7, key=f"mood_{day}")
                caffeine = st.slider(f"Caffeine cups ({day})", 0, 5, 1, key=f"caffeine_{day}")
                tasks = st.slider(f"Tasks planned ({day})", 0, 20, 5, key=f"tasks_{day}")
                church = st.checkbox(f"Church day ({day})", key=f"church_{day}")

        submitted_weekly = st.form_submit_button("Get Weekly Predictions")
        if submitted_weekly:
            for day in days:
                input_dict = {
                    "is_church_day": int(st.session_state[f"church_{day}"]),
                    "wake_up_time_hr": st.session_state[f"wake_{day}"],
                    "sleep_hours": st.session_state[f"sleep_{day}"],
                    "sleep_quality_1to10": st.session_state[f"qual_{day}"],
                    "morning_prayer_done": int(st.session_state[f"prayer_{day}"]),
                    "breakfast_eaten": int(st.session_state[f"breakfast_{day}"]),
                    "exercise_done": int(st.session_state[f"exercise_{day}"]),
                    "exercise_minutes": st.session_state[f"exmin_{day}"],
                    "classes_attended": st.session_state[f"classes_{day}"],
                    "stress_level_1to10": st.session_state[f"stress_{day}"],
                    "mood_morning_1to10": st.session_state[f"mood_{day}"],
                    "caffeine_intake_cups": st.session_state[f"caffeine_{day}"],
                    "tasks_planned": st.session_state[f"tasks_{day}"],
                    "day_of_week": day
                }
                predictions[day] = predict_day(input_dict)

    if predictions:
        st.subheader("Weekly Predictions")
        for day, pred in predictions.items():
            st.write(f"**{day}**: {pred}")
        st.write("These are predictions based on your planned schedules. Adjust as needed!")