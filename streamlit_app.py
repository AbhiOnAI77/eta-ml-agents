import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from geopy.distance import geodesic
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ML and Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Food Delivery ETA Prediction",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4ECDC4;
        font-weight: 600;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF6B6B;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 3rem;
        font-size: 1.1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🚴 Food Delivery ETA Prediction System</p>', unsafe_allow_html=True)
st.markdown("### End-to-End ML Pipeline: Data Generation → Feature Engineering → Model Training → Real-time Prediction")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/scooter.png", width=100)
    st.title("⚙️ Configuration")

    num_records = st.slider("Dataset Size", 1000, 100000, 10000, step=1000)
    test_size = st.slider("Test Split %", 10, 30, 20, step=5) / 100
    val_size = st.slider("Validation Split %", 10, 30, 20, step=5) / 100

    st.markdown("---")
    st.markdown("### 📊 Model Parameters")
    mlp_epochs = st.slider("MLP Epochs", 10, 100, 50, step=10)
    batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)

    run_training = st.button("🚀 Generate Data & Train Models", type="primary")

# Cache data generation
@st.cache_data
def simulate_delivery_data(num_records=10000):
    """Generate synthetic food delivery data"""
    def haversine_distance(lat1, lon1, lat2, lon2):
        return geodesic((lat1, lon1), (lat2, lon2)).km

    pickup_times = [
        datetime.now() - timedelta(days=random.randint(1, 365), minutes=random.randint(1, 60))
        for _ in range(num_records)
    ]

    data = {
        'order_id': [f'ORD_{i}' for i in range(num_records)],
        'rider_id': [f'RIDER_{random.randint(1, 500)}' for _ in range(num_records)],
        'restaurant_lat': np.random.uniform(17.3, 17.5, num_records),
        'restaurant_lon': np.random.uniform(78.3, 78.6, num_records),
        'customer_lat': np.random.uniform(17.3, 17.5, num_records),
        'customer_lon': np.random.uniform(78.3, 78.6, num_records),
        'pickup_time': pickup_times,
        'actual_travel_time_minutes': np.random.normal(30, 10, num_records).clip(5, 60),
        'scheduled_travel_time_minutes': np.random.normal(25, 8, num_records).clip(5, 50),
        'num_items': np.random.randint(1, 10, num_records),
        'order_value_usd': np.random.uniform(5, 50, num_records),
        'restaurant_type': random.choices(['fast_food', 'fine_dining', 'cafe', 'dessert', 'street_food'], k=num_records),
        'weather_conditions': random.choices(['clear', 'rainy', 'cloudy', 'stormy'], weights=[0.5, 0.2, 0.2, 0.1], k=num_records),
        'traffic_level': random.choices(['low', 'moderate', 'high', 'severe'], weights=[0.3, 0.4, 0.2, 0.1], k=num_records),
        'rider_avg_speed_kmph': np.random.normal(20, 5, num_records).clip(10, 40),
        'rider_break_frequency_hr': np.random.normal(0.5, 0.2, num_records).clip(0, 1),
        'rider_acceptance_rate': np.random.uniform(0.7, 1.0, num_records),
        'rider_rating': np.random.uniform(3.0, 5.0, num_records),
        'rider_shift_hours': np.random.uniform(0, 12, num_records)
    }

    df = pd.DataFrame(data)

    # Feature Engineering
    df['pickup_hour'] = pd.to_datetime(df['pickup_time']).dt.hour
    df['pickup_dayofweek'] = pd.to_datetime(df['pickup_time']).dt.dayofweek
    df['pickup_month'] = pd.to_datetime(df['pickup_time']).dt.month
    df['pickup_dayofyear'] = pd.to_datetime(df['pickup_time']).dt.dayofyear

    # Calculate route distance
    df['route_distance_km'] = df.apply(
        lambda row: haversine_distance(
            row['restaurant_lat'], row['restaurant_lon'],
            row['customer_lat'], row['customer_lon']
        ), axis=1
    )

    # Create target: was delivery late?
    df['was_late'] = (df['actual_travel_time_minutes'] > df['scheduled_travel_time_minutes']).astype(int)

    # One-hot encode weather
    weather_dummies = pd.get_dummies(df['weather_conditions'], prefix='weather')
    df = pd.concat([df, weather_dummies], axis=1)

    return df

def prepare_data(df):
    """Prepare data for modeling"""
    # Drop columns not needed for modeling
    features_to_drop = ['order_id', 'rider_id', 'pickup_time', 'restaurant_lat', 'restaurant_lon',
                        'customer_lat', 'customer_lon', 'weather_conditions',
                        'scheduled_travel_time_minutes']

    X = df.drop(columns=features_to_drop + ['actual_travel_time_minutes'])
    y = df['actual_travel_time_minutes']

    return X, y

def train_mlp_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """Train Multi-Layer Perceptron model"""
    mlp_model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1)
    ])

    mlp_model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae']
    )

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = mlp_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[early_stop]
    )

    return mlp_model, history

# Main application logic
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False

if run_training or st.session_state.data_generated:
    if run_training:
        with st.spinner('🔄 Generating synthetic data...'):
            df = simulate_delivery_data(num_records)
            st.session_state.df = df
            st.session_state.data_generated = True
    else:
        df = st.session_state.df

    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview", 
        "🔍 EDA", 
        "🤖 Model Training", 
        "📈 Model Performance",
        "🎯 Real-time Prediction"
    ])

    with tab1:
        st.info("👉 **Next Step:** After reviewing the data, click the '**🔍 EDA**' tab for visual insights.")
        st.markdown('<p class="sub-header">Dataset Statistics</p>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Orders", f"{len(df):,}")
        with col2:
            st.metric("Avg Delivery Time", f"{df['actual_travel_time_minutes'].mean():.1f} min")
        with col3:
            late_rate = (df['was_late'].sum() / len(df)) * 100
            st.metric("Late Delivery Rate", f"{late_rate:.1f}%")
        with col4:
            st.metric("Avg Distance", f"{df['route_distance_km'].mean():.2f} km")

        st.markdown("---")
        st.markdown("### Sample Data")
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("### Feature Statistics")
        st.dataframe(df.describe(), use_container_width=True)

    with tab2:
        st.info("👉 **Next Step:** When you are ready, head to the '**🤖 Model Training**' tab to train our predictive models!")
        st.markdown('<p class="sub-header">Exploratory Data Analysis</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            # Delivery time distribution
            fig = px.histogram(
                df, 
                x='actual_travel_time_minutes',
                nbins=50,
                title='Distribution of Actual Delivery Time',
                labels={'actual_travel_time_minutes': 'Delivery Time (minutes)'}
            )
            fig.update_traces(marker_color='#FF6B6B')
            st.plotly_chart(fig, use_container_width=True)

            # Traffic vs Delivery Time
            fig = px.box(
                df,
                x='traffic_level',
                y='actual_travel_time_minutes',
                title='Delivery Time by Traffic Level',
                color='traffic_level',
                labels={'actual_travel_time_minutes': 'Delivery Time (minutes)'}
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Restaurant type analysis
            restaurant_stats = df.groupby('restaurant_type')['actual_travel_time_minutes'].mean().sort_values()
            fig = px.bar(
                restaurant_stats,
                orientation='h',
                title='Average Delivery Time by Restaurant Type',
                labels={'value': 'Avg Time (minutes)', 'index': 'Restaurant Type'}
            )
            fig.update_traces(marker_color='#4ECDC4')
            st.plotly_chart(fig, use_container_width=True)

            # Weather impact
            fig = px.box(
                df,
                x='weather_conditions',
                y='actual_travel_time_minutes',
                title='Delivery Time by Weather Conditions',
                color='weather_conditions',
                labels={'actual_travel_time_minutes': 'Delivery Time (minutes)'}
            )
            st.plotly_chart(fig, use_container_width=True)

        # Correlation heatmap
        st.markdown("### Feature Correlations with Delivery Time")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_with_target = df[numeric_cols].corr()['actual_travel_time_minutes'].sort_values(ascending=False)

        fig = px.bar(
            x=corr_with_target.values,
            y=corr_with_target.index,
            orientation='h',
            title='Correlation with Actual Delivery Time',
            labels={'x': 'Correlation', 'y': 'Feature'}
        )
        fig.update_traces(marker_color=['#FF6B6B' if x > 0 else '#4ECDC4' for x in corr_with_target.values])
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.info("👉 **Next Step:** Click '**▶️ Train Models**' below. Once training completes, the **📈 Model Performance** tab will have metrics ready.")
        st.markdown('<p class="sub-header">Model Training Pipeline</p>', unsafe_allow_html=True)

        if st.button("▶️ Train Models", type="primary"):
            # Prepare data
            X, y = prepare_data(df)

            # Categorical and numerical features
            categorical_features = ['restaurant_type', 'traffic_level']
            numerical_features = [col for col in X.columns if col not in categorical_features]

            # Create preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_features),
                    ('cat', OrdinalEncoder(), categorical_features)
                ]
            )

            # Train-val-test split
            X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            val_ratio = val_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_ratio, random_state=42)

            # Fit preprocessor and transform
            X_train_processed = preprocessor.fit_transform(X_train)
            X_val_processed = preprocessor.transform(X_val)
            X_test_processed = preprocessor.transform(X_test)

            # Store in session state
            st.session_state.preprocessor = preprocessor
            st.session_state.X_test_processed = X_test_processed
            st.session_state.y_test = y_test
            st.session_state.feature_names = preprocessor.get_feature_names_out()

            col1, col2 = st.columns(2)

            with col1:
                st.info("🌲 Training Random Forest...")
                # Random Forest
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X_train_processed, y_train)
                rf_pred = rf.predict(X_test_processed)

                rf_mae = mean_absolute_error(y_test, rf_pred)
                rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
                rf_r2 = r2_score(y_test, rf_pred)

                st.session_state.rf_model = rf
                st.session_state.rf_metrics = {'MAE': rf_mae, 'RMSE': rf_rmse, 'R²': rf_r2}
                st.session_state.rf_pred = rf_pred

                st.success("✅ Random Forest trained!")
                st.metric("MAE", f"{rf_mae:.2f} min")
                st.metric("RMSE", f"{rf_rmse:.2f} min")
                st.metric("R² Score", f"{rf_r2:.3f}")

            with col2:
                st.info("🧠 Training MLP Neural Network...")
                # MLP
                mlp_model, history = train_mlp_model(
                    X_train_processed, y_train,
                    X_val_processed, y_val,
                    epochs=mlp_epochs,
                    batch_size=batch_size
                )

                mlp_pred = mlp_model.predict(X_test_processed).flatten()
                mlp_mae = mean_absolute_error(y_test, mlp_pred)
                mlp_rmse = np.sqrt(mean_squared_error(y_test, mlp_pred))
                mlp_r2 = r2_score(y_test, mlp_pred)

                st.session_state.mlp_model = mlp_model
                st.session_state.mlp_history = history
                st.session_state.mlp_metrics = {'MAE': mlp_mae, 'RMSE': mlp_rmse, 'R²': mlp_r2}
                st.session_state.mlp_pred = mlp_pred

                st.success("✅ MLP trained!")
                st.metric("MAE", f"{mlp_mae:.2f} min")
                st.metric("RMSE", f"{mlp_rmse:.2f} min")
                st.metric("R² Score", f"{mlp_r2:.3f}")

            # Feature importance
            st.markdown("### 🎯 Feature Importance (Random Forest)")
            importance = rf.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': st.session_state.feature_names,
                'Importance': importance
            }).sort_values(by='Importance', ascending=False).head(15)

            fig = px.bar(
                feature_importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 15 Features by Importance'
            )
            fig.update_traces(marker_color='#FF6B6B')
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.info("👉 **Next Step:** Now that our models are evaluated, try driving it yourself in the '**🎯 Real-time Prediction**' tab!")
        st.markdown('<p class="sub-header">Model Performance Analysis</p>', unsafe_allow_html=True)

        if 'rf_model' in st.session_state and 'mlp_model' in st.session_state:
            # Metrics comparison
            col1, col2, col3 = st.columns(3)

            metrics_df = pd.DataFrame({
                'Model': ['Random Forest', 'MLP Neural Net'],
                'MAE': [st.session_state.rf_metrics['MAE'], st.session_state.mlp_metrics['MAE']],
                'RMSE': [st.session_state.rf_metrics['RMSE'], st.session_state.mlp_metrics['RMSE']],
                'R²': [st.session_state.rf_metrics['R²'], st.session_state.mlp_metrics['R²']]
            })

            st.dataframe(metrics_df, use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                # Actual vs Predicted - Random Forest
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=st.session_state.y_test,
                    y=st.session_state.rf_pred,
                    mode='markers',
                    name='Random Forest',
                    marker=dict(color='#FF6B6B', size=5, opacity=0.6)
                ))
                fig.add_trace(go.Scatter(
                    x=[st.session_state.y_test.min(), st.session_state.y_test.max()],
                    y=[st.session_state.y_test.min(), st.session_state.y_test.max()],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='black', dash='dash')
                ))
                fig.update_layout(
                    title='Random Forest: Actual vs Predicted',
                    xaxis_title='Actual Time (min)',
                    yaxis_title='Predicted Time (min)'
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Actual vs Predicted - MLP
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=st.session_state.y_test,
                    y=st.session_state.mlp_pred,
                    mode='markers',
                    name='MLP',
                    marker=dict(color='#4ECDC4', size=5, opacity=0.6)
                ))
                fig.add_trace(go.Scatter(
                    x=[st.session_state.y_test.min(), st.session_state.y_test.max()],
                    y=[st.session_state.y_test.min(), st.session_state.y_test.max()],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='black', dash='dash')
                ))
                fig.update_layout(
                    title='MLP: Actual vs Predicted',
                    xaxis_title='Actual Time (min)',
                    yaxis_title='Predicted Time (min)'
                )
                st.plotly_chart(fig, use_container_width=True)

            # Training history for MLP
            if 'mlp_history' in st.session_state:
                st.markdown("### MLP Training History")
                history = st.session_state.mlp_history.history

                fig = make_subplots(rows=1, cols=2, subplot_titles=['Loss', 'MAE'])

                fig.add_trace(go.Scatter(y=history['loss'], name='Train Loss', line=dict(color='#FF6B6B')), row=1, col=1)
                fig.add_trace(go.Scatter(y=history['val_loss'], name='Val Loss', line=dict(color='#4ECDC4')), row=1, col=1)

                fig.add_trace(go.Scatter(y=history['mae'], name='Train MAE', line=dict(color='#FF6B6B')), row=1, col=2)
                fig.add_trace(go.Scatter(y=history['val_mae'], name='Val MAE', line=dict(color='#4ECDC4')), row=1, col=2)

                fig.update_xaxes(title_text="Epoch", row=1, col=1)
                fig.update_xaxes(title_text="Epoch", row=1, col=2)
                fig.update_layout(height=400)

                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👈 Please train the models first in the 'Model Training' tab")

    with tab5:
        st.info("👉 **Action:** Adjust sliders and weather conditions for a mock order, then click '**🎯 Predict ETA**' to see the predicted delivery time.")
        st.markdown('<p class="sub-header">Real-time ETA Prediction</p>', unsafe_allow_html=True)

        if 'rf_model' in st.session_state and 'mlp_model' in st.session_state:
            st.markdown("### Enter Order Details")

            col1, col2, col3 = st.columns(3)

            with col1:
                num_items_input = st.number_input("Number of Items", 1, 10, 3)
                order_value_input = st.number_input("Order Value (USD)", 5.0, 100.0, 25.0)
                restaurant_type_input = st.selectbox("Restaurant Type", 
                    ['fast_food', 'fine_dining', 'cafe', 'dessert', 'street_food'])
                traffic_level_input = st.selectbox("Traffic Level", 
                    ['low', 'moderate', 'high', 'severe'])

            with col2:
                rider_speed_input = st.slider("Rider Avg Speed (km/h)", 10.0, 40.0, 20.0)
                rider_break_freq_input = st.slider("Rider Break Frequency (per hour)", 0.0, 1.0, 0.5)
                rider_acceptance_input = st.slider("Rider Acceptance Rate", 0.7, 1.0, 0.85)
                rider_rating_input = st.slider("Rider Rating", 3.0, 5.0, 4.0)

            with col3:
                rider_shift_hours_input = st.slider("Rider Shift Hours", 0.0, 12.0, 6.0)
                pickup_hour_input = st.slider("Pickup Hour", 0, 23, 12)
                route_distance_input = st.number_input("Route Distance (km)", 1.0, 30.0, 10.0)
                was_late_input = st.selectbox("Previous Late Delivery?", [0, 1])

            # Weather checkboxes
            st.markdown("#### Weather Conditions")
            col_w1, col_w2, col_w3, col_w4 = st.columns(4)
            with col_w1:
                weather_clear = st.checkbox("Clear", value=True)
            with col_w2:
                weather_cloudy = st.checkbox("Cloudy", value=False)
            with col_w3:
                weather_rainy = st.checkbox("Rainy", value=False)
            with col_w4:
                weather_stormy = st.checkbox("Stormy", value=False)

            if st.button("🎯 Predict ETA", type="primary"):
                # Create input dataframe
                input_data = {
                    'num_items': num_items_input,
                    'order_value_usd': order_value_input,
                    'restaurant_type': restaurant_type_input,
                    'traffic_level': traffic_level_input,
                    'rider_avg_speed_kmph': rider_speed_input,
                    'rider_break_frequency_hr': rider_break_freq_input,
                    'rider_acceptance_rate': rider_acceptance_input,
                    'rider_rating': rider_rating_input,
                    'rider_shift_hours': rider_shift_hours_input,
                    'pickup_hour': pickup_hour_input,
                    'pickup_dayofweek': datetime.now().weekday(),
                    'pickup_month': datetime.now().month,
                    'pickup_dayofyear': datetime.now().timetuple().tm_yday,
                    'route_distance_km': route_distance_input,
                    'was_late': was_late_input,
                    'weather_clear': weather_clear,
                    'weather_cloudy': weather_cloudy,
                    'weather_rainy': weather_rainy,
                    'weather_stormy': weather_stormy
                }

                input_df = pd.DataFrame([input_data])

                # Preprocess
                input_processed = st.session_state.preprocessor.transform(input_df)

                # Predict
                rf_prediction = st.session_state.rf_model.predict(input_processed)[0]
                mlp_prediction = st.session_state.mlp_model.predict(input_processed).flatten()[0]

                # Display predictions
                st.markdown("---")
                st.markdown("### 📊 Prediction Results")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("#### 🌲 Random Forest")
                    st.markdown(f"<h1 style='text-align: center; color: #FF6B6B;'>{rf_prediction:.1f} min</h1>", 
                              unsafe_allow_html=True)

                with col2:
                    st.markdown("#### 🧠 MLP Neural Net")
                    st.markdown(f"<h1 style='text-align: center; color: #4ECDC4;'>{mlp_prediction:.1f} min</h1>", 
                              unsafe_allow_html=True)

                with col3:
                    avg_prediction = (rf_prediction + mlp_prediction) / 2
                    st.markdown("#### 📈 Ensemble Average")
                    st.markdown(f"<h1 style='text-align: center; color: #95E1D3;'>{avg_prediction:.1f} min</h1>", 
                              unsafe_allow_html=True)

                # Prediction confidence
                prediction_diff = abs(rf_prediction - mlp_prediction)
                if prediction_diff < 2:
                    confidence = "High"
                    color = "green"
                elif prediction_diff < 5:
                    confidence = "Medium"
                    color = "orange"
                else:
                    confidence = "Low"
                    color = "red"

                st.markdown(f"**Prediction Confidence:** :{color}[{confidence}] (Model agreement: {prediction_diff:.1f} min difference)")
        else:
            st.info("👈 Please train the models first in the 'Model Training' tab")

else:
    st.info("👈 **Start Here:** Configure parameters in the sidebar and click '**🚀 Generate Data & Train Models**' to begin!")

    st.markdown("## 🚀 Quick Start Guide")
    st.markdown("""
    Welcome to the Food Delivery ETA Prediction System! Follow these 3 easy steps to explore the app:
    
    1. **⚙️ Configure & Generate**: Open the sidebar on the left, choose your dataset size, and click **🚀 Generate Data & Train Models**. This creates a realistic synthetic dataset.
    2. **📊 Explore & Train**: Once generated, navigate through the tabs to view Data Stats, explore relationships in the EDA tab, and evaluate the performance of our Random Forest & Neural Network models.
    3. **🎯 Predict**: Go to the '**Real-time Prediction**' tab to input custom order details (like traffic, weather, distance) and get an instant, AI-driven ETA prediction!
    """)
    st.markdown("---")

    # Show project overview
    st.markdown("## 🎯 Project Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 📋 Features
        - **Synthetic Data Generation**: Create realistic delivery datasets
        - **Feature Engineering**: Time-based, geospatial, and behavioral features
        - **Multiple Models**: Random Forest & MLP Neural Network
        - **Interactive EDA**: Comprehensive visualizations
        - **Real-time Predictions**: Test with custom inputs
        - **Model Comparison**: Side-by-side performance analysis
        """)

    with col2:
        st.markdown("""
        ### 🛠️ Tech Stack
        - **Frontend**: Streamlit, Plotly
        - **ML**: Scikit-learn, TensorFlow/Keras
        - **Data**: Pandas, NumPy, GeoPy
        - **Models**: Random Forest, Multi-Layer Perceptron

        ### 📊 Key Metrics
        - Mean Absolute Error (MAE)
        - Root Mean Squared Error (RMSE)
        - R² Score
        """)

    st.markdown("---")
    st.markdown("### 💼 Business Impact")
    st.markdown("""
    This system demonstrates:
    - **Operations Optimization**: Predict delivery times accurately to improve logistics
    - **Customer Satisfaction**: Set realistic expectations for delivery windows
    - **Resource Allocation**: Better assignment of riders based on predicted ETAs
    - **Performance Monitoring**: Identify factors contributing to delays
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🚴 Food Delivery ETA Prediction System | Built with Streamlit & TensorFlow</p>
    <p>Showcasing: Data Science • Machine Learning • Deep Learning • MLOps</p>
</div>
""", unsafe_allow_html=True)
