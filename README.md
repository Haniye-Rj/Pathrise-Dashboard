# Pathrise Project – README

**Student:**
Haniyeh Raji
**Student ID:** 488209

---

## Project Description

Pathrise is a career accelerator founded in 2017 in San Francisco. The company was created from the founders’ desire to *systematize* the often random and high-stress job search process for tech professionals. The dataset used in this project contains **2,544 entries and 16 columns**, representing real-world applicant information.

In this project, historical data from Pathrise was used to develop a **machine learning classification model** that predicts whether a new candidate has a realistic chance of finding a job within a timeframe that is considered logical for the company. The target variable of the model is **job placement success**.

In addition to the predictive model, a **dashboard application** was developed for administrators to explore the dataset, analyze feature behavior, and gain business insights. This dashboard is designed to support decision-making and stakeholder reporting. The project demonstrates a complete data pipeline including **data wrangling, machine learning, interactive visualization, secure authentication, and SQL persistence**.

The final solution is implemented as a **Streamlit-based web application** that combines real-time prediction, analytics, and data management functionality.

---

## Dataset

**Original dataset link:**
[https://www.kaggle.com/api/v1/datasets/download/ahmadmakhdoomi/pathrise-dataset?dataset_version_number=1](https://www.kaggle.com/api/v1/datasets/download/ahmadmakhdoomi/pathrise-dataset?dataset_version_number=1)
(Access: anyone with the link can view/download)

**Dataset Explanation: Pathrise Applicants**

The dataset represents real-world applicant data collected by Pathrise.

**Features used in the model:**

* **primary_track (Categorical):** Applicant’s field (SWE, Data, Design, etc.)
* **employment_status (Categorical):** Current work status (Student, Employed, etc.)
* **highest_level_of_education (Categorical):** Degree level (Bachelor’s, Master’s, etc.)
* **work_authorization_status (Categorical):** Visa or citizenship status
* **number_of_applications (Numeric):** Number of jobs applied to
* **number_of_interviews (Numeric):** Number of interviews received
* **professional_experience_num (Numeric):** Years of professional experience
* **program_duration_days (Numeric):** Days spent in the Pathrise program

**Target Variable:**

* **placed:** Binary variable (1 = Placed, 0 = Not Placed)

**Note:**
Dataset file names were kept exactly as in the original download.

---

## How to Run

The project is developed in **two main parts**:

1. **Data preprocessing and model development**, implemented in the `cleaning_Model_dev` file.
2. **Application layer**, implemented in `app.py`.

Follow the steps below to run the project:

1. Download the dataset from the link provided above.
2. Run the `cleaning_Model_dev` file using the downloaded dataset to perform data cleaning, preprocessing, and model training. This step generates the required model artifacts and the cleaned CSV file.
3. To run the application, ensure the following files are present in the root directory:

   * `app.py` – Main Streamlit application
   * `rf_model.pkl` – Trained Random Forest model
   * `scaler.pkl` – StandardScaler object
   * `model_columns.pkl` – List of training features
4. Start the application by running:

   ```bash
   streamlit run app.py
   ```
5. To view the **data dashboard**, upload the cleaned CSV file generated from the preprocessing step into the application.

---

## Development & Implementation Details

### Application Overview

**Pathrise Placement Predictor & Data Dashboard** is a comprehensive Streamlit web application that predicts job placement success using a **Random Forest Classifier**. The system supports real-time inference, advanced analytics, and administrative data exploration.

### Key Features

* **User System (RBAC):**
  Role-based access control with two user levels:

  * **Admin:** Full data access and analytics dashboard
  * **Basic:** Prediction calculator only

* **Secure Authentication:**
  User credentials are stored in **SQLite** with **SHA-256 password hashing**.

* **Machine Learning Predictor:**
  Real-time prediction using a trained Random Forest model on historical Pathrise applicant data.

* **Data Explorer & Dataset Library:**
  Admin users can upload CSV files of historical data used for training and analyze them across multiple tabs.

* **Advanced Analytics:**
  Over 10 interactive **Plotly** visualizations, including heatmaps and success-rate analysis.

* **SQL Integration:**

  * Automated logging of every prediction
  * Relational dataset library for dataset management

### Project Structure & Logic

* **init_db():** Initializes relational tables for users, predictions, and stored datasets
* **Authentication Logic:** Uses `st.session_state` to maintain login status across tabs
* **One-Hot Encoding:** User inputs are manually encoded to match the training feature space
* **Dataset Library:** Uploaded CSV files are stored physically in a `stored_datasets/` folder while metadata is saved in SQLite

---

## Technical Stack

* **Frontend:** Streamlit
* **Machine Learning:** Scikit-Learn (Random Forest, StandardScaler)
* **Database:** SQLite3
* **Visualizations:** Plotly Express
* **Security:** Hashlib (SHA-256)
* **Data Handling:** Pandas, NumPy

---

## Deployed Application

[https://pathrise-dashboard.streamlit.app/](https://pathrise-dashboard.streamlit.app/)
for accessing admin role, Username: admin
Password: admin123
