# FPL AI Manager

**FPL AI Manager** is a fully functional web application that integrates Machine Learning (Random Forest) and Traditional AI (Greedy Optimization) to solve the complex problem of selecting an optimal Fantasy Premier League squad.

The system predicts player performance using historical data and then uses a constraint-satisfaction algorithm to generate the highest-scoring squad possible within a specific budget.

## Features

* **Data Pipeline:** Automated fetching and cleaning of FPL Gameweek data from the 2024-25 season.
* **Machine Learning Model:** A **Random Forest Regressor** trained on player form, influence, and minutes to predict future points (`xP`).
* **Player Calibration:** Implementation of specific heuristic adjustments (e.g., "The Saka Fix") to prevent over-prediction on outliers.
* **Traditional AI Optimization:** A custom **Greedy Algorithm** that solves the Knapsack-style selection problem, adhering to strict FPL rules (Budget, max 3 players per team, specific position counts).
* **Interactive Dashboard:** A Streamlit-based UI to adjust budgets, visualize the pitch, and analyze prediction error (residuals).

---

## Tech Stack

* **Language:** Python 3.11+
* **Frontend:** Streamlit
* **Machine Learning:** Scikit-Learn (RandomForest, KMeans), Joblib
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn

---

## Project Structure
 ```bash
    fpl_manager_agent/
    ├── LICENSE
    ├── README.md               # Project Documentation
    └── Project/
        ├── app.py              # Main Application Entry Point (Streamlit)
        ├── fpl.ipynb           # Model Training & Exploratory Data Analysis Notebook
        ├── data/               # CSV Data storage
        │   ├── final_predictions.csv
        │   ├── mergedgw.csv
        │   └── simulation_predictions.csv
        └── models/
            └── rf_model.pkl    # Serialized Random Forest Model
```

## Setup Instructions
1. Clone the Repository
```bash
git clone https://github.com/umer33511/fpl_manager_agent.git
cd fpl_manager_agent
```
2. Install Dependencies
Create a requirements.txt file (or install manually) with the following packages:
```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn joblib
```
3. Train the Model (First Run)
Before running the app, you must generate the model file (rf_model.pkl).

Open Project/fpl.ipynb in Jupyter Notebook or VS Code.

Run all cells sequentially.

This will:

Fetch the latest data.

Train the Random Forest.

Save the model to Project/models/rf_model.pkl.

4. Run the Application
Navigate to the root directory and run:
```bash
streamlit run Project/app.py
```
The application will open in your default browser at http://localhost:8501.

## API & Logic Documentation

Since this is a Streamlit application, it does not expose REST API endpoints (like GET/POST). Instead, the application logic is modularized into specific Python functions within `app.py`.

### 1. Data Ingestion (`fetch_and_process_data`)
* **Input:** Season string (default: "2024-25").
* **Process:** Loops through Gameweeks 1-38 fetching data from the GitHub repository `vaastav/Fantasy-Premier-League`.
* **Feature Engineering:** Calculates rolling averages for:
    * `form_last_3` (Average points over last 3 games).
    * `avg_minutes_last_3` (Average playtime).
    * `avg_influence_last_3` (Underlying stats).

### 2. ML Prediction (Random Forest)
* **Model Source:** `Project/models/rf_model.pkl`.
* **Features Used:** `['form_last_3', 'avg_minutes_last_3', 'avg_influence_last_3', 'value', 'was_home']`.
* **Output:** Adds a `predicted_points` column to the dataframe.

### 3. Optimization Algorithm (`solve_fpl_strong_xi`)
This corresponds to the **Traditional AI** component. It is a **Greedy Approach** with Heuristic adjustments.

**Algorithmic Workflow:**
1.  **Calibration:** Calculates an `xMins_factor` to penalize players who do not play full 90 minutes.
2.  **Pool Segmentation:** Splits players into "Starters" (Sorted by Predicted Points) and "Fodder" (Sorted by Lowest Cost).
3.  **Phase 1 (Bench Filling):** Selects the cheapest valid players to fill the bench slots (1 GK, 1 DEF, 1 MID, 1 FWD) to save budget.
4.  **Phase 2 (Starter Selection):** Iterates through the top-predicted players.
    * *Constraint Check:* Checks if the player fits position limits (e.g., max 5 DEFs) and team limits (max 3 from one club).
    * *Budget Check:* Checks if `current_cost + player_cost <= total_budget`.
5.  **Captaincy:** Assigns the Captain armband to the selected starter with the highest predicted points.

---

## Application Interface

1.  **Sidebar Settings:**
    * **Gameweek Slider:** Choose which week to simulate.
    * **Budget Slider:** Adjust available funds (default £100m).
2.  **Squad Optimization Tab:**
    * Displays the selected 15-man squad.
    * Visualizes the pitch with starters and bench.
    * Compares **AI Prediction** vs **Actual Points**.
3.  **AI Insights Tab:**
    * **Scatter Plot:** Visualizes the correlation between predicted points and actual results (identifying over/under performers).
    * **Residual Analysis:** A bar chart showing the exact error margin for every player in the squad.

---

## LICENSE

This project is licensed under the MIT License - see the `LICENSE` file for details.
