import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import sqlite3
import joblib
from datetime import datetime
import hashlib
from enum import Enum

# Define roles clearly
class UserRole(Enum):
    ADMIN = "admin"
    BASIC = "basic"

def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def init_db():
    conn = sqlite3.connect('pathrise_data.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    timestamp DATETIME,
                    track TEXT,
                    education TEXT,
                    experience REAL,
                    interviews INTEGER,
                    result TEXT,
                    probability REAL
                )''')
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT, 
                    password TEXT, 
                    role TEXT)''')
    
    admin_pwd = hash_password("admin123")
    c.execute("INSERT OR IGNORE INTO users VALUES (1, 'admin', ?, ?)", 
              (admin_pwd, UserRole.ADMIN.value))
    
    c.execute("INSERT OR IGNORE INTO users VALUES (2, 'user1', ?, ?)", 
              (hash_password("test"), UserRole.BASIC.value))
    
    conn.commit()
    conn.close()

def create_user(username, password):
    conn = sqlite3.connect('pathrise_data.db')
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM users")
    result = cur.fetchone()
    try:
        cur.execute(
            "INSERT INTO users (id, username, password, role) VALUES (?, ?, ?, ?)",
            (result[0] + 1, username, hash_password(password), UserRole.BASIC.value)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

#validation_function
column_names=["primary_track",	"program_duration_days",	"placed",	"employment_status",	"highest_level_of_education",	"biggest_challenge_in_search",	"work_authorization_status",	"number_of_interviews",	"number_of_applications",	"professional_experience_num",]
def validate_csv(df):
    if len(df.columns) !=10:
       st.info("Wrong number of columns")
       return False
    i=0
    for col_name, col_val in df.items():
       if col_name != column_names[i]:
           st.info("Columns name are not as they are defined")
           return False
       i+=1
    return True

# --- Login Logic ---
def login_section():
    if not st.session_state.get('logged_in'):
        st.sidebar.subheader("Credentials")
        user = st.sidebar.text_input("Username")
        pwd = st.sidebar.text_input("Password", type="password")
        
        if st.sidebar.button("Login"):
            hashed_input = hash_password(pwd)
            conn = sqlite3.connect('pathrise_data.db')
            c = conn.cursor()
            c.execute("SELECT id, role FROM users WHERE username=? AND password=?", (user, hashed_input))
            result = c.fetchone()
            conn.close()
            
            if result:
                st.session_state['logged_in'] = True
                st.session_state['user_id'] = result[0]
                st.session_state['username'] = user
                st.session_state['role'] = result[1]
                st.rerun()
            else:
                st.sidebar.error("Invalid credentials")
        if st.sidebar.button("Register"):
            hashed_input = hash_password(pwd)
            conn = sqlite3.connect('pathrise_data.db')
            c = conn.cursor()
            c.execute("SELECT id, role FROM users WHERE username=? AND password=?", (user, hashed_input))
            result = c.fetchone()
            if not user or not pwd:
                st.error("Username and password cannot be empty")
            else:
                ok = create_user(user, pwd)
                if ok:
                    st.success("Account created! You can now log in.")
                else:
                    st.error("Username already exists")
            conn.close()

# ---------------------------------------------------------
# 1. Page Configuration
# ---------------------------------------------------------
st.set_page_config(page_title="Pathrise Dashboard", layout="wide")

# ---------------------------------------------------------
# 2. Load ML Assets
# ---------------------------------------------------------
@st.cache_resource
def load_assets():
    model = joblib.load('rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
    model_columns = joblib.load('model_columns.pkl')
    return model, scaler, model_columns


def run_app():
    model, scaler, model_columns = load_assets()

    uploaded_validated = False
    # -----------------------------
    # Dynamic Tab Generation
    # -----------------------------
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if st.session_state['logged_in']:
        # Define tabs based on role
        if st.session_state['role'] == UserRole.ADMIN.value:
            tabs = st.tabs(["Placement Predictor", "Model Feature Analysis", "Non-Model Contextual Insights", "History"])
        elif st.session_state['role'] == UserRole.BASIC.value:
            tabs = st.tabs(["Placement Predictor"])
            st.info("Basic access: You only have permission to use the Calculator.")
        else:
            st.info("Please log in to access.")
            # ---------------------------------------------------------
        # 3. Sidebar: Persistent Applicant Inputs
        # ---------------------------------------------------------
        st.sidebar.header("New Applicant Info")

        # Categorical Inputs
        employment_status = st.sidebar.selectbox("Employment Status", 
            ["Contractor", "Employed", "Student", "Unemployed"])

        education = st.sidebar.selectbox("Highest Level of Education", 
            ["Bachelor's Degree", "Master's Degree", "Doctorate or Professional Degree", 
            "GED or equivalent", "High School Graduate", "Some College, No Degree", "Some High School"])

        primary_track = st.sidebar.selectbox("Primary Track", 
            ["Data", "Design", "Other", "PSO", "SWE"])

        work_auth = st.sidebar.selectbox("Work Authorization", 
            ["Citizen", "Green Card", "Other"])

        # Numeric Inputs
        applications = st.sidebar.number_input("Number of Applications", min_value=0, max_value=1000, value=10)
        interviews = st.sidebar.number_input("Number of Interviews", min_value=0, max_value=100, value=0)
        experience = st.sidebar.number_input("Professional Experience (Years)", min_value=0.0, max_value=10.0, value=1.0)
        duration = st.sidebar.number_input("Program Duration (Days)", min_value=0.0, max_value=365.0, value=1.0)

        # --- TAB 1: PREDICTOR (Available to ALL) ---
        with tabs[0]:
            st.header("Placement Calculator")
            st.info("Insert your data to predict")
        if st.sidebar.button("Run Prediction"):
            # 1. Create Input DataFrame
            input_df = pd.DataFrame(columns=model_columns)
            input_df.loc[0] = 0  
            
            # 2. Fill Numeric
            input_df['program_duration_days'] = duration
            input_df['number_of_interviews'] = interviews
            input_df['number_of_applications'] = applications
            input_df['professional_experience_num'] = experience
            
            # 3. One-Hot Encoding (using f-strings to match column names)
            input_df[f"primary_track_{primary_track}"] = 1
            input_df[f"highest_level_of_education_{education}"] = 1
            input_df[f"employment_status_{employment_status}"] = 1
            input_df[f"work_authorization_status_{work_auth}"] = 1
            
            # 4. Scale
            num_cols = ["program_duration_days", "number_of_interviews", "number_of_applications", "professional_experience_num"]
            input_df[num_cols] = scaler.transform(input_df[num_cols])

            # 5. Result
            pred = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]

            result_text = "Placed" if pred == 1 else "Not Placed"

            # Connect and Insert
            conn = sqlite3.connect('pathrise_data.db')
            c = conn.cursor()
            c.execute('''INSERT INTO prediction_history 
                        (timestamp, user_id, track, education, experience, interviews, result, probability)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', 
                    (datetime.now(), st.session_state['user_id'], primary_track, education, experience, interviews, result_text, prob))
            conn.commit()
            conn.close()

            st.subheader("Prediction Result")
            if pred == 1:
                st.success(f"**PLACED** (Probability: {prob:.2%})")
            else:
                st.error(f"**NOT PLACED** (Probability: {1-prob:.2%})")

        if st.sidebar.button("Logout"):
            st.session_state["logged_in"] = False
            st.session_state["username"] = None
            st.rerun()
            
        # --- TAB 2 & 3: ADMIN ONLY ---
        if st.session_state['role'] == 'admin':
            with tabs[1]:
                st.header("Data Explorer")
                uploaded = st.file_uploader("Upload your CSV file for EDA", key="eda_upload")

                if uploaded:
                    df = pd.read_csv(uploaded, encoding='latin1')
                    uploaded_validated = validate_csv(df)
                    if uploaded_validated:
                        sub_overview, sub_type, sub_visuals = st.tabs(["Overview", "Types", "Visualizations"])

                        with sub_overview:
                            st.markdown("### Data")
                            st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
                            st.dataframe(df)
                            
                            st.markdown("### Numeric Summary")
                            st.dataframe(df.describe())

                        with sub_type:
                                st.write("**Data Types**")
                                st.dataframe(df.dtypes.to_frame("Dtype"))

                        with sub_visuals:
                                st.header("Categorical Distribution Analysis")

                                col_left, col_right = st.columns(2)

                                with col_left:
                                    st.subheader("Employment Status")
                                    emp_counts = df['employment_status'].value_counts().reset_index()
                                    emp_counts.columns = ['Status', 'Count']
                                    
                                    # Pie Chart 
                                    fig_pie = px.pie(
                                        emp_counts, 
                                        values='Count', 
                                        names='Status', 
                                        color_discrete_sequence=['#2e8b57', '#3cb371', '#66cdaa', '#98fb98'],
                                        hole=0.3 # Optional: makes it a donut chart
                                    )
                                    st.plotly_chart(fig_pie, use_container_width=True)

                                with col_right:
                                    st.subheader("Primary Track")
                                    track_counts = df['primary_track'].value_counts().reset_index()
                                    track_counts.columns = ['Track', 'Count']
                                    
                                    # Bar Chart
                                    fig_bar = px.bar(
                                        track_counts, 
                                        x='Track', 
                                        y='Count',
                                        color='Track',
                                        color_discrete_sequence=['#2e8b57', '#3cb371', '#66cdaa', '#98fb98']
                                    )
                                    st.plotly_chart(fig_bar, use_container_width=True)

                                    # 3. WORK AUTHORIZATION (Full Width)
                                    st.divider()
                                    st.subheader("Work Authorization Overview")
                                    auth_counts = df['work_authorization_status'].value_counts().reset_index()
                                    auth_counts.columns = ['Authorization', 'Count']
                                    
                                    fig_auth = px.bar(
                                        auth_counts, 
                                        y='Authorization', 
                                        x='Count', 
                                        orientation='h',
                                        color_discrete_sequence=['#2e8b57']
                                    )
                                    st.plotly_chart(fig_auth, use_container_width=True)

            with tabs[2]:
                st.header("Insights Beyond the Model")
                st.write("These visualizations analyze variables not directly used to train the Random Forest.")

                if uploaded_validated:
                    st.subheader("The Path to Success: Education & Track")
                    fig5 = px.sunburst(df, path=['highest_level_of_education', 'primary_track', 'placed'],
                                    color='placed', color_discrete_map={'(?)':'black', '0':'#ef553b', '1':'#2e8b57'})
                    fig5.update_layout(margin=dict(t=40, l=0, r=0, b=0))
                    st.plotly_chart(fig5, use_container_width=True)
                    
                    st.subheader("How long do candidates stay in Pathrise?")
                    fig7 = px.histogram(df, x="program_duration_days", color="placed",
                                        marginal="rug", nbins=30,
                                        color_discrete_sequence=['#ef553b', '#2e8b57'])
                    st.plotly_chart(fig7, use_container_width=True)

                    st.subheader("Top Challenges for Job Seekers")

                    challenge_counts = df['biggest_challenge_in_search'].value_counts().reset_index()
                    challenge_counts.columns = ['Challenge', 'Count']

                    fig7 = px.bar(
                    challenge_counts, 
                    x='Count', 
                    y='Challenge', 
                    orientation='h',
                    title="What is stopping our applicants?",
                    color='Count',
                    color_continuous_scale='Greens'
                    )
                    fig7.update_layout(yaxis={'categoryorder':'total ascending'}) 
                    st.plotly_chart(fig7, use_container_width=True)

                    st.subheader("Placement Success Rate by Challenge Type")

                    # Calculate placement rate (%) for each challenge
                    challenge_success = df.groupby('biggest_challenge_in_search')['placed'].mean().reset_index()
                    challenge_success['Placement Rate (%)'] = challenge_success['placed'] * 100

                            # A professional "Forest and Sea" palette with high visibility
                    custom_greens = ['#004d40', '#1b5e20', '#2e7d32', '#388e3c', '#43a047', '#66bb6a']

                    fig7 = px.bar(
                        challenge_counts, 
                        x='Count', 
                        y='Challenge', 
                        orientation='h',
                        title="What is stopping our applicants?",
                        color='Challenge', # Color by category instead of a gradient
                        color_discrete_sequence=custom_greens 
                    )

                    fig9 = px.bar(
                        challenge_success.sort_values('Placement Rate (%)', ascending=False),
                        x='biggest_challenge_in_search',
                        y='Placement Rate (%)',
                        color='Placement Rate (%)',
                        color_continuous_scale='RdYlGn', 
                        title="Which challenges actually prevent placement?"
                    )
                    st.plotly_chart(fig9, use_container_width=True)

                    st.subheader("Hurdle Heatmap: Track vs. Challenge")

                    # Create a pivot table of counts
                    pivot_df = df.groupby(['primary_track', 'biggest_challenge_in_search']).size().unstack(fill_value=0)

                    fig10 = px.imshow(
                        pivot_df,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale='Greens',
                        title="Do different tracks face different challenges?"
                    )
                    st.plotly_chart(fig10, use_container_width=True)

                else:
                    st.info("Upload data in the previous tab to see these insights.")

            with tabs[3]:
                st.header("Global Prediction History")
                st.write("View all predictions made by all users.")
                conn = sqlite3.connect('pathrise_data.db')
                # Query EVERYTHING (No WHERE clause)
                all_history = pd.read_sql("SELECT * FROM prediction_history", conn)
                st.dataframe(all_history)
                
                # Admin-only business metric: Total predictions made today
                total_preds = len(all_history)
                st.metric("Total System Predictions", total_preds)
                st.subheader("Average Success Probability per Education Level")
                agg_query = "SELECT education, AVG(probability) as avg_success FROM prediction_history GROUP BY education"
                agg_df = pd.read_sql(agg_query, conn)
                
                # Plotting the SQL result with Plotly (using dark greens for visibility)
                fig_sql = px.bar(agg_df, x='education', y='avg_success', color_discrete_sequence=['#004d40'])
                st.plotly_chart(fig_sql)
                conn.close()
            
    else:
        st.warning("Please log in or register to access the application.")
            
def main():
    init_db()
    login_section()
    run_app()

if __name__ == "__main__":
    main()
      
