
# Pathrise Placement Predictor & Data Dashboard 

A comprehensive Streamlit-based web application that predicts job placement success for Pathrise applicants using **Random Forest Classifier**. The tool features a secure login system with tiered access, interactive Plotly visualizations, a SQL-backed dataset library, and real-time inference.

## Project Overview

This project serves as a career-tech tool to help job seekers understand their placement probability based on historical data. It demonstrates a full data pipeline: **Wrangling → SQL Persistence → Machine Learning → Interactive Presentation**.

### Key Features

* **Tiered User System (RBAC):** Distinct privileges for `Admin` (full data access) and `Basic` (calculator only) users.
* **Secure Authentication:** User accounts stored in SQLite with **SHA-256 password hashing**.
* **ML Predictor:** Real-time prediction using a Random Forest model trained on Pathrise applicant data.
* **Data Explorer & Library:** Users can upload CSVs, save them to a server-side library, and analyze them across multiple tabs.
* **Advanced Analytics:** 10+ interactive Plotly charts, including Sunburst, Heatmaps, and success-rate analysis.
* **SQL Integration:** Automated logging of every prediction and a relational library for dataset management.

---

## Dataset Explanation: Pathrise Applicants

The dataset used to train the model represents real-world applicant data from Pathrise, a career accelerator.

### Features Used in the Model:

| Feature | Type | Description |
| --- | --- | --- |
| `primary_track` | Categorical | The applicant's field (SWE, Data, Design, etc.). |
| `employment_status` | Categorical | Current work status (Student, Employed, etc.). |
| `highest_level_of_education` | Categorical | The user's degree (Bachelor's, Master's, etc.). |
| `work_authorization_status` | Categorical | Visa/Citizenship status. |
| `number_of_applications` | Numeric | Count of jobs applied to. |
| `number_of_interviews` | Numeric | Count of interviews secured. |
| `professional_experience_num` | Numeric | Years of relevant professional experience. |
| `program_duration_days` | Numeric | Days spent in the Pathrise program. |

**Target Variable:**

* `placed`: A binary indicator (1 = Placed, 0 = Not Placed).

---

## Technical Stack

* **Frontend:** [Streamlit](https://streamlit.io/)
* **Machine Learning:** Scikit-Learn (Random Forest, StandardScaler)
* **Database:** SQLite3
* **Visualizations:** Plotly Express, Seaborn, Matplotlib
* **Security:** Hashlib (SHA-256)
* **Data Handling:** Pandas, NumPy, OS

### 3. Files Required

Ensure the following files are in the root directory:

* `app.py` (The main script)
* `rf_model.pkl` (Trained model)
* `scaler.pkl` (StandardScaler object)
* `model_columns.pkl` (List of training features)


## Project Structure & Logic

* **`init_db()`**: Sets up the relational database for users, predictions, and the dataset library.
* **Authentication Logic**: Uses `st.session_state` to maintain login status across different tabs.
* **One-Hot Encoding**: The app manually creates dummy variables for user input to match the model's training format.
* **Dataset Library**: Saves physical `.csv` files to a `stored_datasets/` folder while keeping a record in SQL.

