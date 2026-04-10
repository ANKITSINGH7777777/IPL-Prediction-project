🏏 IPL Match Prediction using Machine Learning
📌 Problem Statement

Predict the winning probability of IPL teams during a live match using machine learning techniques. The goal is to build a model that can estimate match outcomes based on real-time match conditions like runs, wickets, overs, and target.
🚀 Features
Predict winning probability of both teams in real-time
Interactive UI using Streamlit
Clean and simple user interface
Uses trained ML model for prediction
Displays probability in percentage format
Supports multiple IPL teams
⚙️ Tech Stack
Python 🐍
Pandas & NumPy
Scikit-learn
Streamlit
Pickle (for model saving)
📂 Project Structure
IPL-Prediction-Project/
│
├── app.py
├── model.pkl
├── dataset.csv
├── requirements.txt
└── README.md
🛠️ Setup Instructions
1. Clone the repository
git clone https://github.com/your-username/IPL-Prediction-project.git
cd IPL-Prediction-project
2. Install dependencies
pip install -r requirements.txt
3. Run the app
streamlit run app.py
📊 Dataset
The dataset contains historical IPL match data
Features include:
Batting team
Bowling team
City
Runs
Wickets
Overs
Target

👉 Dataset Source:
(👉 Add your dataset link here, e.g. Kaggle or GitHub)

📸 Screenshots
🔹 Home Page

(Add screenshot here)

🔹 Prediction Output

(Add screenshot here)

📈 Model Details
Algorithm used: Logistic Regression / Random Forest (update based on your model)
Trained on historical IPL data
Outputs probability instead of just win/loss
🌐 Live Demo

👉 https://ipl-prediction-project-1.onrender.com/

🙌 Future Improvements
Add more advanced models (XGBoost, Deep Learning)
Include player-level stats
Improve UI/UX
Add match history visualization
