import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, date
from PIL import Image
from streamlit_extras.app_logo import add_logo

# Mock data generation for pitching
def generate_mock_pitching_data(num_pitchers=10, num_days=100):
    pitchers = [f"Pitcher {i}" for i in range(1, num_pitchers + 1)]
    dates = pd.date_range(end=date.today(), periods=num_days)
    pitch_types = ['Fastball', '2 Seam', 'Sinker', 'Cutter', 'Change Up', 'Splitter', 'Slider', 'Curveball']
    data = []
    
    for pitcher in pitchers:
        for d in dates:
            for pitch in pitch_types:
                data.append({
                    'pitcher': pitcher,
                    'date': d,
                    'pitch_type': pitch,
                    'velo': np.random.normal(90, 5),
                    'workload_compliance': np.random.uniform(0.8, 1),
                    'miss_distance': np.random.normal(10, 2),
                    'stuff_plus': np.random.normal(100, 10),
                    'level': np.random.choice(['Youth', 'High School', 'College', 'Professional']),
                    'gym': np.random.choice(['WA', 'AZ', 'FL']),
                    'trainer': np.random.choice(['Trainer A', 'Trainer B', 'Trainer C'])
                })
    
    return pd.DataFrame(data)

# Pitcher Trends Page
def pitcher_trends(df):
    st.header("Pitcher Trends")
    
    # Sidebar
    pitcher = st.sidebar.selectbox("Select Pitcher", df['pitcher'].unique())
    pitch_type = st.sidebar.multiselect("Select Pitch Type", df['pitch_type'].unique(), default=df['pitch_type'].unique())
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    start_date = st.sidebar.date_input("Start Date", min_date)
    end_date = st.sidebar.date_input("End Date", max_date)
    
    # Filter data
    pitcher_data = df[(df['pitcher'] == pitcher) & 
                      (df['pitch_type'].isin(pitch_type)) &
                      (df['date'].dt.date >= start_date) & 
                      (df['date'].dt.date <= end_date)]
    
    # Key Metrics
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Velo", f"{pitcher_data['velo'].mean():.2f}")
    col2.metric("Workload Compliance", f"{pitcher_data['workload_compliance'].mean():.2%}")
    col3.metric("Avg Miss Distance", f"{pitcher_data['miss_distance'].mean():.2f}")
    col4.metric("Avg Stuff+", f"{pitcher_data['stuff_plus'].mean():.2f}")
    
    # Metric Graphs and Gains/Losses
    st.subheader("Metric Trends and Gains/Losses")
    metrics = ['velo', 'workload_compliance', 'miss_distance', 'stuff_plus']
    selected_metrics = st.multiselect("Select metrics to display", metrics, default=['velo'])
    
    # Create two columns
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Metric Graphs
        for metric in selected_metrics:
            fig = px.line(pitcher_data, x='date', y=metric, color='pitch_type', title=f"{metric.replace('_', ' ').title()} Over Time")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Gains/Losses (mock data)
        st.subheader("Gains/Losses")
        gains_losses = pd.DataFrame({
            'Metric': metrics,
            'Gain/Loss': np.random.uniform(-5, 5, len(metrics)),
            'Expected': np.random.uniform(-2, 2, len(metrics))
        })
        st.dataframe(gains_losses, hide_index=True, use_container_width=True)
    
    # Data Table
    st.subheader("Data Table")
    show_data = st.checkbox("Show Data Table", value=False)
    if show_data:
        st.dataframe(pitcher_data)
    else:
        st.info("Check the box above to display the data table.")

# In-Gym Trends Page
def in_gym_trends(df):
    st.header("In-Gym Trends")
    
    # Sidebar
    level = st.sidebar.multiselect("Select Level", ['Youth', 'High School', 'College', 'Professional'], default=['High School'])
    gym = st.sidebar.multiselect("Select Gym", ['WA', 'AZ', 'FL'], default=['WA', 'AZ', 'FL'])
    pitch_type = st.sidebar.multiselect("Select Pitch Type", df['pitch_type'].unique(), default=df['pitch_type'].unique())
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    start_date = st.sidebar.date_input("Start Date", min_date)
    end_date = st.sidebar.date_input("End Date", max_date)
    
    # Filter data
    gym_data = df[(df['level'].isin(level)) & 
                  (df['gym'].isin(gym)) &
                  (df['pitch_type'].isin(pitch_type)) &
                  (df['date'].dt.date >= start_date) & 
                  (df['date'].dt.date <= end_date)]
    
    # Overall Gym Performance
    st.subheader("Overall Gym Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Velo", f"{gym_data['velo'].mean():.2f}")
    col2.metric("Avg Workload Compliance", f"{gym_data['workload_compliance'].mean():.2%}")
    col3.metric("Avg Miss Distance", f"{gym_data['miss_distance'].mean():.2f}")
    col4.metric("Avg Stuff+", f"{gym_data['stuff_plus'].mean():.2f}")
    
    # Leaderboards
    st.subheader("Leaderboards")
    leaderboard_metric = st.selectbox("Select Leaderboard", ['velo', 'workload_compliance', 'miss_distance', 'stuff_plus'])
    leaderboard = gym_data.groupby('pitcher')[leaderboard_metric].mean().sort_values(ascending=False).head(10)
    st.bar_chart(leaderboard)
    
    # Trend Graphs
    st.subheader("Gym-wide Trends")
    metrics = ['velo', 'workload_compliance', 'miss_distance', 'stuff_plus']
    fig = go.Figure()
    for metric in metrics:
        trend = gym_data.groupby('date')[metric].mean()
        fig.add_trace(go.Scatter(x=trend.index, y=trend.values, mode='lines', name=metric))
    fig.update_layout(title="Gym-wide Trends Over Time", xaxis_title="Date", yaxis_title="Value")
    st.plotly_chart(fig)

    # Gym Comparison
    st.subheader("Gym Comparison")
    comparison_metric = st.selectbox("Select Metric for Gym Comparison", metrics)
    gym_comparison = gym_data.groupby('gym')[comparison_metric].mean().sort_values(ascending=False)
    st.bar_chart(gym_comparison)

# Trainer Trends Page
def trainer_trends(df):
    st.header("Trainer Performance Comparison")
    
    # Sidebar
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    start_date = st.sidebar.date_input("Start Date", min_date)
    end_date = st.sidebar.date_input("End Date", max_date)
    pitch_type = st.sidebar.multiselect("Select Pitch Type", df['pitch_type'].unique(), default=df['pitch_type'].unique())
    
    # Level filter in sidebar
    use_level_filter = st.sidebar.checkbox("Filter by Athlete Level", value=False)
    if use_level_filter:
        levels = st.sidebar.multiselect("Select Athlete Levels", ['Youth', 'High School', 'College', 'Professional'], default=['High School', 'College'])
    else:
        levels = ['Youth', 'High School', 'College', 'Professional']
    
    # Filter data by date, level, and pitch type
    df_filtered = df[(df['date'].dt.date >= start_date) & 
                     (df['date'].dt.date <= end_date) & 
                     (df['level'].isin(levels)) &
                     (df['pitch_type'].isin(pitch_type))]
    
    # Individual Athlete Improvement
    st.subheader("Individual Athlete Improvement")
    selected_trainer = st.selectbox("Select Trainer", df_filtered['trainer'].unique())
    trainer_data = df_filtered[df_filtered['trainer'] == selected_trainer]
    
    metric = st.selectbox("Select metric", ['velo', 'workload_compliance', 'miss_distance', 'stuff_plus'])
    improvement = trainer_data.groupby(['pitcher', 'date'])[metric].mean().unstack(level=0)
    fig = px.line(improvement, x=improvement.index, y=improvement.columns, title=f"Athlete {metric.replace('_', ' ').title()} Improvement")
    st.plotly_chart(fig)
    
    # Calculate overall averages
    overall_avg = df_filtered[['velo', 'workload_compliance', 'miss_distance', 'stuff_plus']].mean()
    
    # Calculate trainer averages
    trainer_avg = df_filtered.groupby('trainer')[['velo', 'workload_compliance', 'miss_distance', 'stuff_plus']].mean()
    
    # Calculate percentage difference from overall average
    trainer_performance = (trainer_avg - overall_avg) / overall_avg * 100
    
    # Visualize trainer performance
    st.subheader("Trainer Performance Visualization")
    
    fig = go.Figure()
    for trainer in trainer_performance.index:
        fig.add_trace(go.Bar(
            x=[trainer],
            y=[trainer_performance.loc[trainer, metric]],
            name=trainer
        ))
    
    fig.update_layout(
        title=f"{metric.replace('_', ' ').title()} Performance by Trainer",
        yaxis_title="% Difference from Average",
        xaxis_title="Trainer"
    )
    st.plotly_chart(fig)
    
    # Expandable section for Trainer Performance % and Best Performing Trainers
    with st.expander("**Trainer Performance Analysis Tab**"):
        # Display trainer performance
        st.subheader("Trainer Performance (% difference from overall average)")
        st.dataframe(trainer_performance.style.format("{:.2f}%"))
        
        # Identify best performing trainer for each metric
        best_trainers = trainer_performance.idxmax()
        st.subheader("Best Performing Trainers")
        for metric, trainer in best_trainers.items():
            st.write(f"{metric.replace('_', ' ').title()}: {trainer} (+{trainer_performance.loc[trainer, metric]:.2f}%)")
    
    # Trainer Performance by Athlete Level
    st.subheader("Trainer Performance by Athlete Level")
    level_metric = st.selectbox("Select metric for level comparison", ['velo', 'workload_compliance', 'miss_distance', 'stuff_plus'], key='level_metric')
    trainer_level_performance = df_filtered.groupby(['trainer', 'level'])[level_metric].mean().unstack(level=1)
    st.bar_chart(trainer_level_performance)

# Main app
def main():
    st.set_page_config(page_title="Pitching KPI Dashboard", layout="wide")

    # Add the main logo using streamlit_extras
    add_logo("/Users/rohitkrishnan/Desktop/Driveline/logo.png", height=65)

    # Add additional images to the sidebar
    st.sidebar.image("/Users/rohitkrishnan/Desktop/Driveline/logo.png", caption="Chicks hate the long ball")
    
    st.sidebar.title("Navigation")
    
    # Page selection
    page = st.sidebar.radio("Select Page", ["Pitcher Trends", "In-Gym Trends", "Trainer Trends"])
    
    # Main content area title
    st.title("Pitching KPI Dashboard")
    
    # Generate mock data
    df = generate_mock_pitching_data()
    
    # Ensure 'date' column is datetime
    df['date'] = pd.to_datetime(df['date'])
    
    if page == "Pitcher Trends":
        pitcher_trends(df)
    elif page == "In-Gym Trends":
        in_gym_trends(df)
    elif page == "Trainer Trends":
        trainer_trends(df)

if __name__ == "__main__":
    main()