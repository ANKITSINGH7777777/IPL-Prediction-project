import streamlit as st
import pickle
import pandas as pd
import numpy as np # Import numpy for np.inf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import os # Import the os module

# Construct the path to pipe.pkl relative to the current script
script_dir = os.path.dirname(__file__)
pipe_path = os.path.join(script_dir, 'pipe.pkl')

# Load the trained pipeline
try:
    pipe=pickle.load(open(pipe_path,'rb')) # Use the constructed path
    st.success('Model pipeline loaded successfully!') # Add this line to verify
except FileNotFoundError:
    st.error(f"Error: '{pipe_path}' not found. Please ensure the model file is in the same directory as 'app.py'.")
    st.stop() # Stop the app if the model can't be loaded

# Define teams and cities (from previous cells)
teams=[
    'Mumbai Indians',
    'Chennai Super Kings',
    'Kolkata Knight Riders',
    'Royal Challengers Bangalore',
    'Delhi Capitals',
    'Sunrisers Hyderabad',
    'Punjab Kings',
    'Lucknow Super Giants',
    'Rajasthan Royals',
    'Gujarat Titans'
]
cities=[
    'Bangalore', 'Delhi', 'Mumbai', 'Chennai', 'Kolkata', 'Jaipur',
    'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion',
    'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein',
    'Ahmedabad', 'Pune', 'Hyderabad', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Cuttack', 'Visakhapatnam', 'Bengaluru', 'Dubai', 'Sharjah', 'Navi Mumbai',
    'Chandigarh', 'Lucknow', 'Guwahati', 'Dharamsala', 'Mohali'
]

# Streamlit App UI
st.set_page_config(layout="centered") # Set page layout
st.title('🏏 IPL Win Predictor')
st.markdown('<p style="font-size:20px;">Predict the winning chances of your favorite IPL team!</p>', unsafe_allow_html=True)

# Sidebar for match setup
st.sidebar.header('Match Setup')
batting_team = st.sidebar.selectbox('Select the Batting Team', sorted(teams))
bowling_team = st.sidebar.selectbox('Select the Bowling Team', sorted(teams))
selected_city = st.sidebar.selectbox('Select Host City', sorted(cities))
target = st.sidebar.number_input('Target Score', min_value=0, max_value=400, value=150)

# Main area for live match progress
st.header('Live Match Progress')

col1, col2, col3 = st.columns(3)
with col1:
    score = st.number_input('Current Score', min_value=0, max_value=target-1)
with col2:
    over = st.number_input('Overs Completed', min_value=0.0, max_value=19.5, step=0.1, format="%.1f")
with col3:
    wickets = st.number_input('Wickets Fallen', min_value=0, max_value=9)

# Prediction button
st.markdown("---") # Separator
if st.button('Predict Win Probability'):
    if batting_team == bowling_team:
        st.error("Batting and Bowling teams cannot be the same!")
    else:
        runs_left = target - score
        # Calculate balls left, ensuring it's not negative and handling decimal overs
        total_balls_bowled = int(over) * 6 + round((over - int(over)) * 10)
        balls_left = max(0, 120 - total_balls_bowled)

        wickets_remaining = 10 - wickets

        crr = score / over if over > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        # Create input DataFrame for prediction
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_remaining],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        # Handle cases where rrr or crr might be inf or NaN after calculations
        input_df.replace([np.inf, -np.inf], 0, inplace=True) # Replace inf with 0
        input_df.fillna(0, inplace=True) # Replace NaN with 0

        # Predict probability
        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]

        st.subheader("Match Prediction:")
        st.write(f"The **{batting_team}** needs **{runs_left}** runs from **{balls_left}** balls to win with **{wickets_remaining}** wickets remaining.")

        win_probability_str = f"**{batting_team}**: <span style='font-size:24px; color:green;'>{round(win * 100)}%</span>"
        loss_probability_str = f"**{bowling_team}**: <span style='font-size:24px; color:red;'>{round(loss * 100)}%</span>"

        col_win, col_loss = st.columns(2)
        with col_win:
            st.markdown(win_probability_str, unsafe_allow_html=True)
        with col_loss:
            st.markdown(loss_probability_str, unsafe_allow_html=True)

        st.balloons() # Add some celebration if prediction is made
