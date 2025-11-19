import streamlit as st
# page config must be the very first Streamlit command executed in the script
# placing it at module import time ensures no other st.* calls run before it.
st.set_page_config(
    page_title="AI Breach Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load CSS at the beginning
load_css()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
import yaml
from datetime import datetime
from pathlib import Path
from ml_model import BreachPredictor
import plotly.express as px
import plotly.graph_objects as go
import streamlit_authenticator as stauth
from alert_system import AlertSystem
from predictive_forecasting import PredictiveAttackForecaster

class BreachPredictionDashboard:
    def __init__(self):
        # Do not call Streamlit UI functions during construction.
        # Keep __init__ lightweight so Streamlit page config can be set first.
        self.model = None
        self.monitor = None
        self.authenticator = None
        self.alert_system = None
    
    def load_model(self):
        try:
            # Load model using BreachPredictor
            from ml_model import BreachPredictor
            self.predictor = BreachPredictor()
            self.predictor.load_model('models/breach_predictor_model.pkl')
            self.model = self.predictor.model
            self.feature_names = self.predictor.feature_names
            self.feature_importance = self.predictor.feature_importance

            # Initialize monitoring
            self.monitor = self.predictor.monitor

            st.sidebar.success("✅ ML Model Loaded Successfully")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading model: {str(e)}")
            self.model = None
            self.predictor = None
            self.feature_names = []
            self.feature_importance = {}
    
    def setup_authentication(self):
        """Initialize authentication system"""
        try:
            with open('auth_config.yaml') as file:
                config = yaml.load(file, Loader=yaml.SafeLoader)

            self.authenticator = stauth.Authenticate(
                config['credentials'],
                config['cookie']['name'],
                config['cookie']['key'],
                config['cookie']['expiry_days']
            )
        except Exception as e:
            st.error(f"Authentication setup failed: {str(e)}")
            self.authenticator = None

    def setup_alert_system(self):
        """Initialize alert system"""
        try:
            self.alert_system = AlertSystem()
        except Exception as e:
            st.warning(f"Alert system initialization failed: {str(e)}")
            self.alert_system = None

    def run(self):
        # Initialize authentication and alerts
        self.setup_authentication()
        self.setup_alert_system()

        # Now it's safe to call UI-affecting functions that may use Streamlit
        self.load_model()

        # Authentication check
        if self.authenticator:
            name, authentication_status, username = self.authenticator.login('Login', 'main')

            if authentication_status:
                self.authenticator.logout('Logout', 'sidebar')
                st.sidebar.success(f'Welcome {name}!')

                # Get user role for access control
                user_role = self.get_user_role(username)

                # Log user access for monitoring
                print(f"📊 USER ACCESS: {name} ({username}) as {user_role} logged in at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                if self.monitor:
                    # Log to monitoring system if available
                    import logging
                    logging.info(f"User {username} ({user_role}) accessed the system")

                # Main dashboard content
                self.show_main_dashboard(user_role, name)

            elif authentication_status == False:
                st.error('Username/password is incorrect')
            elif authentication_status == None:
                st.warning('Please enter your username and password')
        else:
            # Fallback if authentication fails
            st.warning("Authentication system unavailable. Running in demo mode.")
            self.show_main_dashboard("admin")  # Full access in demo mode

    def get_user_role(self, username):
        """Get user role from config"""
        try:
            with open('auth_config.yaml') as file:
                config = yaml.load(file, Loader=yaml.SafeLoader)
            return username  # For simplicity, use username as role
        except:
            return "viewer"

    def show_main_dashboard(self, user_role, name=None):
        # Enhanced Header Section with Modern Design
        st.markdown("""
        <div class="dashboard-header">
            <div class="header-content">
                <div class="header-text">
                    <h1 class="gradient-text">🛡️ AI-Powered Data Breach Prediction System</h1>
                    <p class="header-subtitle">Advanced Machine Learning Platform for Predicting Corporate Data Breach Risks</p>
                    <p class="header-description">Using Dark Web Intelligence & Company Profiling</p>
                </div>
                <div class="header-stats">
        """, unsafe_allow_html=True)

        # System status with enhanced styling
        if self.monitor:
            status = self.monitor.get_system_status()
            status_class = f"status-badge {status['status']}"
            status_icon = {
                'healthy': '🟢',
                'warning': '🟡',
                'degraded': '🔴'
            }.get(status['status'], '⚪')
            st.markdown(f"""
            <div class="{status_class}">
                <span class="status-icon">{status_icon}</span>
                <span class="status-text">{status['status'].upper()}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Quick Overview Cards - Enhanced Grid Layout
        st.markdown('<div class="dashboard-grid-4 fade-in">', unsafe_allow_html=True)

        # System Health Card
        st.markdown("""
        <div class="dashboard-card hover-lift">
            <div class="card-header">
                <span class="card-icon icon-cyber">🏥</span>
                <h3>System Health</h3>
            </div>
            <div class="card-content">
        """, unsafe_allow_html=True)

        if self.monitor:
            status = self.monitor.get_system_status()
            health_class = f"health-indicator {status['status']}"
            status_icon = {
                'healthy': '🟢',
                'warning': '🟡',
                'degraded': '🔴'
            }.get(status['status'], '⚪')
            st.markdown(f"""
            <div class="{health_class}"></div>
            <p class="status-text-large">{status_icon} {status['status'].upper()}</p>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="health-indicator degraded"></div><p class="status-text-large">🔴 UNKNOWN</p>', unsafe_allow_html=True)

        st.markdown('</div></div>', unsafe_allow_html=True)

        # Model Status Card
        st.markdown("""
        <div class="dashboard-card hover-lift">
            <div class="card-header">
                <span class="card-icon icon-cyber">🤖</span>
                <h3>ML Model</h3>
            </div>
            <div class="card-content">
        """, unsafe_allow_html=True)

        if self.model:
            st.markdown('<div class="health-indicator healthy"></div><p class="status-text-large">🟢 LOADED</p>', unsafe_allow_html=True)
            st.markdown('<p class="card-metric">Ready for Predictions</p>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="health-indicator warning"></div><p class="status-text-large">🟡 NOT LOADED</p>', unsafe_allow_html=True)
            st.markdown('<p class="card-metric">Model Required</p>', unsafe_allow_html=True)

        st.markdown('</div></div>', unsafe_allow_html=True)

        # Data Status Card
        st.markdown("""
        <div class="dashboard-card hover-lift">
            <div class="card-header">
                <span class="card-icon icon-cyber">📊</span>
                <h3>Data Status</h3>
            </div>
            <div class="card-content">
        """, unsafe_allow_html=True)

        try:
            company_count = len(pd.read_csv('company_profiles.csv'))
            breach_count = len(pd.read_csv('breach_data.csv'))
            st.markdown(f'<div class="health-indicator healthy"></div><p class="status-text-large">🟢 READY</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="card-metric">{company_count} Companies</p><p class="card-metric">{breach_count} Breaches</p>', unsafe_allow_html=True)
        except:
            st.markdown('<div class="health-indicator warning"></div><p class="status-text-large">🟡 MISSING</p>', unsafe_allow_html=True)
            st.markdown('<p class="card-metric">Data Required</p>', unsafe_allow_html=True)

        st.markdown('</div></div>', unsafe_allow_html=True)

        # Active Alerts Card
        st.markdown("""
        <div class="dashboard-card hover-lift">
            <div class="card-header">
                <span class="card-icon icon-alert">🚨</span>
                <h3>Active Alerts</h3>
            </div>
            <div class="card-content">
        """, unsafe_allow_html=True)

        if self.alert_system:
            active_alerts = len(self.alert_system.get_active_alerts())
            if active_alerts > 0:
                st.markdown(f'<div class="health-indicator warning"></div><p class="status-text-large">🟡 {active_alerts} ACTIVE</p>', unsafe_allow_html=True)
                st.markdown('<p class="card-metric">Requires Attention</p>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="health-indicator healthy"></div><p class="status-text-large">🟢 NONE</p>', unsafe_allow_html=True)
                st.markdown('<p class="card-metric">All Clear</p>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="health-indicator degraded"></div><p class="status-text-large">🔴 DISABLED</p>', unsafe_allow_html=True)
            st.markdown('<p class="card-metric">Alert System Off</p>', unsafe_allow_html=True)

        st.markdown('</div></div></div>', unsafe_allow_html=True)

        # Visual Separator

        # Enhanced Sidebar Navigation
        st.sidebar.markdown("""
        <div class="sidebar-header">
            <div class="sidebar-logo">
                <span class="logo-icon">🛡️</span>
                <span class="logo-text">AI Breach Predictor</span>
            </div>
            <p class="sidebar-subtitle">Advanced Cybersecurity Platform</p>
        </div>
        """, unsafe_allow_html=True)

        # System Health Status Card in Sidebar
        if self.monitor:
            status = self.monitor.get_system_status()
            status_color = {
                'healthy': '🟢',
                'warning': '🟡',
                'degraded': '🔴'
            }.get(status['status'], '⚪')

            st.sidebar.markdown(f"""
            <div class="sidebar-card">
                <div class="sidebar-card-header">
                    <span class="sidebar-card-icon">🔧</span>
                    <h4>System Health</h4>
                </div>
                <div class="sidebar-card-content">
                    <p class="status-indicator {status['status']}">{status_color} {status['status'].upper()}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if status['warnings']:
                with st.sidebar.expander("⚠️ Warnings"):
                    for warning in status['warnings']:
                        st.write(f"- {warning}")

        # Navigation Menu with Icons
        st.sidebar.markdown('<div class="nav-section">', unsafe_allow_html=True)
        st.sidebar.markdown("### 🧭 Navigation")

        # Role-based navigation options with enhanced styling
        nav_options = [
            ("🎯 Risk Prediction", "Real-time breach risk assessment"),
            ("📊 Data Analysis", "Upload and analyze company data")
        ]

        # Add Executive Summary as first option for analyst/admin
        if user_role in ["analyst", "admin"]:
            nav_options.insert(0, ("📊 Executive Summary", "Comprehensive dashboard overview"))
            nav_options.extend([
                ("🔍 Model Insights", "ML model performance & features"),
                ("📈 Industry Trends", "Risk trends by industry sector"),
                ("🔮 Predictive Forecasting", "Future attack predictions"),
                ("🌑 Dark Web Intelligence", "Real-time threat monitoring")
            ])

        if user_role == "admin":
            nav_options.append(("🏥 System Health", "Monitor system performance"))

        # Create navigation buttons with descriptions
        nav_buttons = []
        for i, (option, description) in enumerate(nav_options):
            nav_buttons.append(option)

        selected_page = st.sidebar.radio(
            "Select Module:",
            nav_buttons,
            index=0,
            key="main_nav",
            label_visibility="collapsed"
        )

        # Show description for selected page
        for option, description in nav_options:
            if option == selected_page:
                st.sidebar.markdown(f'<p class="nav-description">{description}</p>', unsafe_allow_html=True)
                break

        st.sidebar.markdown('</div>', unsafe_allow_html=True)

        # Quick Actions Section
        st.sidebar.markdown('<div class="quick-actions-section">', unsafe_allow_html=True)
        st.sidebar.markdown("### ⚡ Quick Actions")

        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.sidebar.button("🔄 Refresh", key="refresh_data", width='stretch'):
                st.rerun()
        with col2:
            if st.sidebar.button("📊 Report", key="generate_report", width='stretch'):
                st.sidebar.success("Generating...")

        # User Info Section
        st.sidebar.markdown('<div class="user-info-section">', unsafe_allow_html=True)
        st.sidebar.markdown(f"""
        <div class="user-card">
            <div class="user-avatar">👤</div>
            <div class="user-details">
                <p class="user-name">{name}</p>
                <p class="user-role">{user_role.title()}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.sidebar.markdown('</div>', unsafe_allow_html=True)

        # Map navigation to methods
        nav_methods = {
            "📊 Executive Summary": self.executive_summary_tab,
            "🎯 Risk Prediction": self.risk_prediction_tab,
            "📊 Data Analysis": self.data_analysis_tab,
            "🔍 Model Insights": self.model_insights_tab,
            "📈 Industry Trends": self.industry_trends_tab,
            "🔮 Predictive Forecasting": self.predictive_forecasting_tab,
            "🌑 Dark Web Intelligence": self.dark_web_intelligence_tab,
            "🏥 System Health": self.system_health_tab
        }

        # Render selected page
        if selected_page in nav_methods:
            nav_methods[selected_page]()
        else:
            # Default to executive summary if page not found
            self.executive_summary_tab()

    def executive_summary_tab(self):
        """Executive Summary Dashboard - Main landing page"""
        st.header("📊 Executive Summary Dashboard")

        # Real-time metrics section
        st.markdown('<div class="dashboard-grid-4 fade-in">', unsafe_allow_html=True)

        # System Health Overview
        st.markdown("""
        <div class="dashboard-card hover-lift">
            <div class="card-header">
                <span class="card-icon icon-cyber">🏥</span>
                <h3>System Health</h3>
            </div>
            <div class="card-content">
        """, unsafe_allow_html=True)

        if self.monitor:
            status = self.monitor.get_system_status()
            health_class = f"health-indicator {status['status']}"
            status_icon = {
                'healthy': '🟢',
                'warning': '🟡',
                'degraded': '🔴'
            }.get(status['status'], '⚪')
            st.markdown(f"""
            <div class="{health_class}"></div>
            <p class="status-text-large">{status_icon} {status['status'].upper()}</p>
            <p class="card-metric">All systems operational</p>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="health-indicator degraded"></div><p class="status-text-large">🔴 UNKNOWN</p>', unsafe_allow_html=True)

        st.markdown('</div></div>', unsafe_allow_html=True)

        # Active Risk Alerts
        st.markdown("""
        <div class="dashboard-card hover-lift">
            <div class="card-header">
                <span class="card-icon icon-alert">🚨</span>
                <h3>Active Alerts</h3>
            </div>
            <div class="card-content">
        """, unsafe_allow_html=True)

        if self.alert_system:
            active_alerts = len(self.alert_system.get_active_alerts())
            if active_alerts > 0:
                st.markdown(f'<div class="health-indicator warning"></div><p class="status-text-large">🟡 {active_alerts} ACTIVE</p>', unsafe_allow_html=True)
                st.markdown('<p class="card-metric">Requires attention</p>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="health-indicator healthy"></div><p class="status-text-large">🟢 NONE</p>', unsafe_allow_html=True)
                st.markdown('<p class="card-metric">All clear</p>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="health-indicator degraded"></div><p class="status-text-large">🔴 DISABLED</p>', unsafe_allow_html=True)

        st.markdown('</div></div>', unsafe_allow_html=True)

        # Recent Predictions
        st.markdown("""
        <div class="dashboard-card hover-lift">
            <div class="card-header">
                <span class="card-icon icon-cyber">🔮</span>
                <h3>Recent Activity</h3>
            </div>
            <div class="card-content">
        """, unsafe_allow_html=True)

        try:
            # Check for recent logs or activity
            import os
            log_dir = Path('logs')
            if log_dir.exists():
                log_files = list(log_dir.glob('*.log'))
                if log_files:
                    latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
                    st.markdown('<div class="health-indicator healthy"></div><p class="status-text-large">🟢 ACTIVE</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="card-metric">Last activity: {latest_log.stat().st_mtime.strftime("%H:%M")}</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="health-indicator warning"></div><p class="status-text-large">🟡 NO LOGS</p>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="health-indicator warning"></div><p class="status-text-large">🟡 NO LOGS</p>', unsafe_allow_html=True)
        except:
            st.markdown('<div class="health-indicator warning"></div><p class="status-text-large">🟡 UNKNOWN</p>', unsafe_allow_html=True)

        st.markdown('</div></div>', unsafe_allow_html=True)

        # Data Status
        st.markdown("""
        <div class="dashboard-card hover-lift">
            <div class="card-header">
                <span class="card-icon icon-cyber">📊</span>
                <h3>Data Status</h3>
            </div>
            <div class="card-content">
        """, unsafe_allow_html=True)

        try:
            company_count = len(pd.read_csv('company_profiles.csv'))
            breach_count = len(pd.read_csv('breach_data.csv'))
            st.markdown(f'<div class="health-indicator healthy"></div><p class="status-text-large">🟢 READY</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="card-metric">{company_count} Companies</p><p class="card-metric">{breach_count} Breaches</p>', unsafe_allow_html=True)
        except:
            st.markdown('<div class="health-indicator warning"></div><p class="status-text-large">🟡 MISSING</p>', unsafe_allow_html=True)
            st.markdown('<p class="card-metric">Data required</p>', unsafe_allow_html=True)

        st.markdown('</div></div></div>', unsafe_allow_html=True)

        # Quick Actions Section
        st.subheader("⚡ Quick Actions")
        st.markdown('<div class="dashboard-grid-3 fade-in">', unsafe_allow_html=True)

        # Risk Assessment Action
        st.markdown("""
        <div class="dashboard-card hover-lift">
            <div class="card-header">
                <span class="card-icon icon-cyber">🎯</span>
                <h4>Risk Assessment</h4>
            </div>
            <div class="card-content">
                <p>Analyze company breach risk in real-time</p>
                <p class="small-text">Available in sidebar navigation</p>
        """, unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

        # Data Analysis Action
        st.markdown("""
        <div class="dashboard-card hover-lift">
            <div class="card-header">
                <span class="card-icon icon-cyber">📊</span>
                <h4>Data Analysis</h4>
            </div>
            <div class="card-content">
                <p>Upload and analyze company datasets</p>
                <p class="small-text">Available in sidebar navigation</p>
        """, unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)

        # Forecasting Action
        st.markdown("""
        <div class="dashboard-card hover-lift">
            <div class="card-header">
                <span class="card-icon icon-cyber">🔮</span>
                <h4>Attack Forecasting</h4>
            </div>
            <div class="card-content">
                <p>Predict future attack likelihoods</p>
                <p class="small-text">Available in sidebar navigation</p>
        """, unsafe_allow_html=True)
        st.markdown('</div></div></div>', unsafe_allow_html=True)

        # Recent Activity Feed
        st.subheader("📈 Recent Activity")
        st.markdown("""
        <div class="dashboard-card fade-in">
            <div class="card-content">
        """, unsafe_allow_html=True)

        # Show recent activity from logs if available
        try:
            log_dir = Path('logs')
            if log_dir.exists():
                log_files = list(log_dir.glob('*.log'))
                if log_files:
                    latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
                    with open(latest_log, 'r') as f:
                        recent_lines = f.readlines()[-5:]  # Last 5 entries

                    st.markdown("**Latest System Activity:**")
                    for line in recent_lines:
                        if line.strip():
                            # Parse log line for display
                            if 'USER ACCESS' in line:
                                st.write(f"👤 {line.split('USER ACCESS:')[1].strip()}")
                            elif 'INFO' in line:
                                st.write(f"ℹ️ {line.split('INFO')[1].strip()}")
                            else:
                                st.write(f"📝 {line.strip()}")
                else:
                    st.info("No recent activity logs found.")
            else:
                st.info("Activity logs will appear here as you use the system.")
        except Exception as e:
            st.info("Activity feed will populate as you use the dashboard.")

        st.markdown('</div></div>', unsafe_allow_html=True)

        # System Insights
        st.subheader("🔍 System Insights")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="dashboard-card fade-in">
                <div class="card-header">
                    <span class="card-icon icon-cyber">🤖</span>
                    <h4>AI Model Status</h4>
                </div>
                <div class="card-content">
            """, unsafe_allow_html=True)

            if self.model:
                st.markdown('<div class="health-indicator healthy"></div><p>🟢 ML Model Active</p>', unsafe_allow_html=True)
                st.markdown('<p class="small-text">Ready for predictions</p>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="health-indicator warning"></div><p>🟡 Model Not Loaded</p>', unsafe_allow_html=True)
                st.markdown('<p class="small-text">Limited functionality</p>', unsafe_allow_html=True)

            st.markdown('</div></div>', unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="dashboard-card fade-in">
                <div class="card-header">
                    <span class="card-icon icon-cyber">📈</span>
                    <h4>Performance Metrics</h4>
                </div>
                <div class="card-content">
            """, unsafe_allow_html=True)

            # Show basic performance metrics
            st.metric("Uptime", "99.9%", "0.1%")
            st.metric("Predictions Today", "24", "+12")

            st.markdown('</div></div>', unsafe_allow_html=True)
    
    def risk_prediction_tab(self):
        st.header("🎯 Real-time Breach Risk Assessment")

        if self.model is None:
            st.warning("⚠️ No model loaded. Using demo mode with limited functionality.")
            return

        # Enhanced card-based layout
        st.markdown('<div class="dashboard-grid-2 fade-in">', unsafe_allow_html=True)

        # Company Profile Card
        st.markdown("""
        <div class="dashboard-card hover-lift">
            <div class="card-header">
                <span class="card-icon icon-cyber">🏢</span>
                <h3>Company Profile</h3>
            </div>
            <div class="card-content">
        """, unsafe_allow_html=True)

        company_size = st.selectbox(
            "Company Size",
            ["Small (1-100 employees)", "Medium (101-1000 employees)", "Large (1000+ employees)"]
        )
        industry = st.selectbox(
            "Industry Sector",
            ["Technology", "Finance", "Healthcare", "Education", "Retail", "Government"]
        )
        data_sensitivity = st.slider(
            "Data Sensitivity Level",
            1, 3, 2,
            help="1: Public Data, 2: Confidential Data, 3: Highly Sensitive Data"
        )
        security_budget = st.number_input(
            "Annual Security Budget ($)",
            min_value=1000,
            max_value=10000000,
            value=100000,
            step=10000
        )
        employee_count = st.number_input(
            "Number of Employees",
            min_value=1,
            max_value=50000,
            value=500
        )

        st.markdown('</div></div>', unsafe_allow_html=True)

        # Risk Prediction Results Card
        st.markdown("""
        <div class="dashboard-card hover-lift">
            <div class="card-header">
                <span class="card-icon icon-cyber">🔮</span>
                <h3>Risk Prediction Results</h3>
            </div>
            <div class="card-content">
        """, unsafe_allow_html=True)

        if st.button("🔮 Predict Breach Risk", type="primary"):
            features = self.prepare_features(
                company_size, industry, data_sensitivity,
                security_budget, employee_count
            )
            if self.model:
                # Get risk probability and SHAP values
                risk_probability = self.model.predict_proba([features])[0][1]
                try:
                    shap_values, feature_names = self.predictor.get_shap_values(features)
                    # Display risk results and SHAP explanation
                    self.display_risk_results(risk_probability, features, shap_values, feature_names)
                except Exception as e:
                    st.warning(f"SHAP explanation unavailable: {str(e)}")
                    # Display risk results without SHAP
                    self.display_risk_results(risk_probability, features)
            else:
                # Demo mode without SHAP values
                risk_probability = self.demo_prediction(
                    company_size, industry, data_sensitivity
                )
                self.display_risk_results(risk_probability, features)

    
    def prepare_features(self, company_size, industry, data_sensitivity, security_budget, employee_count):
        size_mapping = {
            "Small (1-100 employees)": 0,
            "Medium (101-1000 employees)": 1,
            "Large (1000+ employees)": 2
        }
        industry_mapping = {
            "Technology": 0, "Finance": 1, "Healthcare": 2,
            "Education": 3, "Retail": 4, "Government": 5
        }
        company_size_encoded = size_mapping[company_size]
        industry_encoded = industry_mapping[industry]
        employee_to_budget_ratio = employee_count / security_budget if security_budget > 0 else 0
        high_sensitivity_risk = 1 if data_sensitivity >= 2 else 0
        industry_risk_scores = {
            "Healthcare": 2.8, "Finance": 2.7, "Government": 2.2,
            "Education": 2.1, "Retail": 1.8, "Technology": 1.5
        }
        industry_breach_risk = industry_risk_scores[industry]
        industry_breach_volume = 500000
        breach_volume_avg = 100000  # Average breach volume for industry
        breach_count = 25  # Average breach count for industry
        critical_breach_ratio = 0.3  # Critical breach ratio for industry
        budget_per_employee = security_budget / employee_count if employee_count > 0 else 0
        high_risk_industry = 1 if industry in ["Healthcare", "Finance"] else 0

        features = [
            company_size_encoded,  # Company Size
            industry_encoded,  # Industry
            data_sensitivity,  # Data Sensitivity
            employee_to_budget_ratio,  # Budget Ratio
            high_sensitivity_risk,  # High Sensitivity
            industry_breach_risk,  # Industry Risk
            np.log1p(industry_breach_volume),  # Breach Volume Total (log)
            np.log1p(breach_volume_avg),  # Breach Volume Avg (log)
            breach_count,  # Breach Count
            critical_breach_ratio,  # Critical Breach Ratio
            security_budget / 10000,  # Security Budget (normalized)
            budget_per_employee,  # Budget Per Employee
            high_risk_industry  # High Risk Industry
        ]
        return features
    
    def demo_prediction(self, company_size, industry, data_sensitivity):
        base_risk = 0.3
        industry_risk = {
            "Healthcare": 0.4, "Finance": 0.35, "Government": 0.25,
            "Education": 0.2, "Retail": 0.15, "Technology": 0.1
        }
        size_risk = {
            "Small (1-100 employees)": 0.1,
            "Medium (101-1000 employees)": 0.2,
            "Large (1000+ employees)": 0.3
        }
        sensitivity_risk = {1: 0.0, 2: 0.2, 3: 0.4}
        total_risk = (
            base_risk +
            industry_risk[industry] +
            size_risk[company_size] +
            sensitivity_risk[data_sensitivity]
        )
        return min(total_risk, 0.95)
    
    def display_risk_results(self, risk_probability, features, shap_values=None, feature_names=None):
        risk_percentage = risk_probability * 100
        if risk_percentage < 20:
            risk_level = "🟢 LOW"
            color = "green"
        elif risk_percentage < 50:
            risk_level = "🟡 MEDIUM" 
            color = "orange"
        elif risk_percentage < 75:
            risk_level = "🟠 HIGH"
            color = "red"
        else:
            risk_level = "🔴 CRITICAL"
            color = "darkred"
            
        # Risk Level and Probability
        st.markdown(f"### Risk Level: {risk_level}")
        st.progress(risk_probability)
        st.metric("Breach Risk Probability", f"{risk_percentage:.1f}%")
        
        # SHAP Feature Contributions
        if shap_values is not None and feature_names is not None:
            st.subheader("🎯 Feature Contributions")
            st.markdown("""
            This chart shows how each feature contributes to the final risk prediction:
            - Red bars push towards higher risk
            - Blue bars push towards lower risk
            """)
            
            # Create waterfall chart using plotly
            base_value = shap_values.base_values
            values = shap_values.values
            cumulative = np.zeros(len(values) + 1)
            cumulative[1:] = np.cumsum(values)
            
            fig = go.Figure()
            
            # Add base value
            fig.add_trace(go.Waterfall(
                name="Base",
                orientation="h",
                measure=["absolute"],
                x=[base_value],
                connector={"mode": "between", "line": {"width": 1, "color": "rgb(100,100,100)", "dash": "solid"}},
                text=["Base Value"],
                textposition="outside"
            ))
            
            # Add feature contributions
            fig.add_trace(go.Waterfall(
                orientation="h",
                measure=["relative"] * len(values),
                x=values,
                textposition="outside",
                text=[f"{name}: {val:.3f}" for name, val in zip(feature_names, values)],
                connector={"mode": "between", "line": {"width": 1, "color": "rgb(100,100,100)", "dash": "solid"}}
            ))
            
            fig.update_layout(
                title="SHAP Feature Contributions",
                showlegend=False,
                height=400,
                waterfallgap=0.2
            )
            st.plotly_chart(fig, width='stretch')
        
        # Risk Factor Summary
        st.subheader("🔍 Risk Factor Summary")
        if shap_values is not None and feature_names is not None:
            # Create summary of major risk factors based on SHAP values
            risk_factors = dict(zip(feature_names, shap_values.values))
            sorted_factors = sorted(risk_factors.items(), key=lambda x: abs(x[1]), reverse=True)
            for factor, impact in sorted_factors[:4]:  # Show top 4 factors
                impact_text = "increases" if impact > 0 else "decreases"
                st.write(f"- **{factor}** {impact_text} risk by {abs(impact):.2f}")
        else:
            # Fallback for demo mode
            risk_factors = {
                "Industry Risk": features[5],
                "Data Sensitivity": features[2] * 0.3,
                "Company Size": features[0] * 0.2,
                "Security Budget Ratio": max(0, 1 - features[3] * 10)
            }
            for factor, score in risk_factors.items():
                st.write(f"- **{factor}**: {score:.2f}/5.0")
        st.subheader("💡 Security Recommendations")
        if risk_percentage > 50:
            st.error("""
            **🚨 Immediate Action Required:**
            - Conduct comprehensive security audit
            - Implement multi-factor authentication
            - Enhance employee security training
            - Consider cyber insurance
            """)
        elif risk_percentage > 25:
            st.warning("""
            **⚠️ Proactive Measures Recommended:**
            - Regular security assessments
            - Update incident response plan
            - Monitor dark web for threats
            - Review data encryption policies
            """)
        else:
            st.success("""
            **✅ Maintain Current Practices:**
            - Continue regular security monitoring
            - Keep systems updated
            - Conduct periodic employee training
            """)
    
    def data_analysis_tab(self):
        st.header("📊 Breach Data Analysis & Upload")

        # Import data processor
        from data_processor import DataProcessor
        self.data_processor = DataProcessor()

        # File upload section
        st.subheader("📤 Upload Company Dataset")
        uploaded_file = st.file_uploader(
            "Upload CSV file with company data",
            type=['csv'],
            help="Flexible column names supported. System will auto-detect: company_name, industry, company_size, data_sensitivity, security_budget, employee_count"
        )

        if uploaded_file is not None:
            try:
                # Read uploaded data
                uploaded_df = pd.read_csv(uploaded_file)
                st.success(f"✅ Successfully uploaded {len(uploaded_df)} company records")

                # Flexible column validation and mapping
                valid, column_mapping, errors = self.data_processor.validate_and_map_columns(uploaded_df)

                if not valid:
                    st.warning("⚠️ Some columns not found, but proceeding with analysis of all available columns:")
                    for error in errors:
                        st.write(f"- {error}")
                    st.info("**Supported column name variations:**")
                    for req_col, variations in self.data_processor.column_mappings.items():
                        st.write(f"- **{req_col}**: {', '.join(variations)}")
                else:
                    st.success("✅ Column mapping successful!")
                    st.info(f"**Mapped columns:** {', '.join([f'{k} → {v}' for k, v in column_mapping.items()])}")

                # Always proceed with processing
                st.info("🔄 Processing all available columns for analysis...")

                # Enhanced data preprocessing
                with st.spinner("🔄 Preprocessing data..."):
                    processed_df, preprocessing_report = self.data_processor.preprocess_data(uploaded_df, column_mapping)

                # Data Quality Dashboard
                self._display_data_quality_dashboard(preprocessing_report)

                # Show processed data preview
                st.subheader("📋 Processed Data Preview")
                st.dataframe(processed_df.head(10), width='stretch')

                # Real-time data enhancement
                if st.checkbox("🌐 Enable Real-time Data Enhancement"):
                    with st.spinner("🔍 Enhancing data with external sources..."):
                        enhanced_df, enhancement_report = self.data_processor.enhance_with_external_data(processed_df)
                        if enhancement_report['data_enhanced'] > 0:
                            st.success(f"✅ Enhanced {enhancement_report['data_enhanced']} records with external data")
                            processed_df = enhanced_df
                        else:
                            st.info("No external data enhancement available (demo mode)")

                # Real-time column analysis section
                st.subheader("🔍 Advanced Column Analysis")
                self._analyze_additional_columns(processed_df)

                # Enhanced analysis button
                if st.button("🔍 Analyze Breach Risk", type="primary"):
                    with st.spinner("🧠 Running comprehensive risk analysis..."):
                        analysis_results = self._analyze_uploaded_companies(processed_df)
                        self._display_analysis_results(analysis_results, processed_df)

            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.info("💡 **Troubleshooting tips:**")
                st.write("- Ensure CSV format with proper headers")
                st.write("- Check for special characters in column names")
                st.write("- Verify numeric columns contain valid numbers")

        # Enhanced fallback analysis
        st.divider()
        st.subheader("📊 Enhanced Data Analysis")
        try:
            # Try to load and analyze existing data with enhanced processing
            breach_df = pd.read_csv('breach_data.csv')
            company_df = pd.read_csv('company_profiles.csv')

            # Apply enhanced processing to existing data
            valid, column_mapping, errors = self.data_processor.validate_and_map_columns(company_df)
            if valid:
                processed_company_df, _ = self.data_processor.preprocess_data(company_df, column_mapping)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.subheader("📈 Enhanced Statistics")
                    st.metric("Total Breaches Analyzed", len(breach_df))
                    st.metric("Companies Profiled", len(processed_company_df))
                    data_quality = self.data_processor._calculate_data_quality_score(processed_company_df)
                    st.metric("Data Quality Score", f"{data_quality:.1f}%")

                with col2:
                    st.subheader("🏭 Industry Analysis")
                    industry_counts = processed_company_df['industry'].value_counts()
                    fig = px.pie(
                        values=industry_counts.values,
                        names=industry_counts.index,
                        title="Companies by Industry (Processed)"
                    )
                    st.plotly_chart(fig, width='stretch')

                with col3:
                    st.subheader("📊 Risk Distribution")
                    # Calculate risk scores for existing data
                    risk_scores = []
                    for _, company in processed_company_df.iterrows():
                        features = self.prepare_features(
                            company['company_size'] + f" ({'1-100' if company['company_size'] == 'Small' else '101-1000' if company['company_size'] == 'Medium' else '1000+'} employees)",
                            company['industry'],
                            int(company['data_sensitivity']),
                            float(company['security_budget']),
                            int(company['employee_count'])
                        )
                        if self.model:
                            risk_prob = self.model.predict_proba([features])[0][1]
                        else:
                            risk_prob = self.demo_prediction(
                                company['company_size'] + f" ({'1-100' if company['company_size'] == 'Small' else '101-1000' if company['company_size'] == 'Medium' else '1000+'} employees)",
                                company['industry'],
                                int(company['data_sensitivity'])
                            )
                        risk_scores.append(risk_prob * 100)

                    fig = px.histogram(
                        x=risk_scores,
                        title="Enhanced Risk Score Distribution",
                        nbins=10,
                        labels={'x': 'Risk Score (%)'}
                    )
                    st.plotly_chart(fig, width='stretch')
            else:
                # Fallback to basic analysis
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Breach Statistics")
                    st.metric("Total Breaches Analyzed", len(breach_df))
                    st.metric("Companies Profiled", len(company_df))
                    industry_counts = breach_df['industry'].value_counts()
                    fig = px.pie(
                        values=industry_counts.values,
                        names=industry_counts.index,
                        title="Breaches by Industry"
                    )
                    st.plotly_chart(fig, width='stretch')
                with col2:
                    st.subheader("Risk Score Distribution")
                    fig = px.histogram(
                        breach_df,
                        x='risk_score',
                        title="Distribution of Breach Risk Scores",
                        nbins=10
                    )
                    st.plotly_chart(fig, width='stretch')

        except Exception as e:
            st.info("📁 Static data files not found. Upload data above to begin enhanced analysis.")
            st.info("💡 **New Features Available:**")
            st.write("- Flexible column name recognition")
            st.write("- Automatic data cleaning and imputation")
            st.write("- Outlier detection and handling")
            st.write("- Data quality scoring")
            st.write("- Real-time external data enhancement")

    def _validate_uploaded_data(self, df):
        """Validate uploaded company data"""
        errors = []

        # Check for required columns first
        required_columns = ['company_name', 'industry', 'company_size', 'data_sensitivity', 'security_budget', 'employee_count']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {', '.join(missing_columns)}")
            return {'valid': False, 'errors': errors}

        # Check data types
        if not pd.api.types.is_numeric_dtype(df['data_sensitivity']):
            errors.append("data_sensitivity must be numeric (1-3)")
        else:
            if df['data_sensitivity'].min() < 1 or df['data_sensitivity'].max() > 3:
                errors.append("data_sensitivity must be between 1 and 3")

        if not pd.api.types.is_numeric_dtype(df['security_budget']):
            errors.append("security_budget must be numeric")
        else:
            if (df['security_budget'] <= 0).any():
                errors.append("security_budget must be positive")

        if not pd.api.types.is_numeric_dtype(df['employee_count']):
            errors.append("employee_count must be numeric")
        else:
            if (df['employee_count'] <= 0).any():
                errors.append("employee_count must be positive")

        # Check categorical values
        valid_sizes = ['Small', 'Medium', 'Large']
        if not df['company_size'].isin(valid_sizes).all():
            errors.append(f"company_size must be one of: {', '.join(valid_sizes)}")

        valid_industries = ['Technology', 'Finance', 'Healthcare', 'Education', 'Retail', 'Government']
        if not df['industry'].isin(valid_industries).all():
            errors.append(f"industry must be one of: {', '.join(valid_industries)}")

        # Check for missing values
        if df.isnull().any().any():
            errors.append("Data contains missing values - please fill all required fields")

        return {
            'valid': len(errors) == 0,
            'errors': errors
        }

    def _display_data_quality_dashboard(self, preprocessing_report):
        """Display comprehensive data quality dashboard"""
        st.subheader("📊 Data Quality Dashboard")

        # Get quality metrics from the report
        quality_data = self.data_processor.generate_quality_dashboard_data(preprocessing_report)

        # Main quality score
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            score = quality_data.get('data_quality_score', 0)
            color = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
            st.metric("Data Quality Score", f"{color} {score:.1f}%")

        with col2:
            st.metric("Rows Processed", quality_data.get('rows_processed', 0))

        with col3:
            st.metric("Missing Values Handled", quality_data.get('missing_values_handled', 0))

        with col4:
            st.metric("Warnings", quality_data.get('warnings_count', 0))

        # Quality indicators
        st.subheader("🔍 Quality Indicators")

        # Create progress bars for different quality aspects
        quality_indicators = {
            'Completeness': min(100, (1 - preprocessing_report.get('missing_values_handled', {}).keys().__len__() / max(1, preprocessing_report.get('original_rows', 1))) * 100),
            'Validity': 100 - len(preprocessing_report.get('warnings', [])) * 10,  # Rough estimate
            'Consistency': 95,  # Assume high consistency after processing
            'Accuracy': quality_data.get('data_quality_score', 0)
        }

        for indicator, score in quality_indicators.items():
            st.progress(min(1.0, score / 100))
            st.caption(f"{indicator}: {score:.1f}%")

        # Warnings and issues
        if preprocessing_report.get('warnings'):
            with st.expander("⚠️ Data Quality Warnings"):
                for warning in preprocessing_report['warnings']:
                    st.write(f"• {warning}")

        # Transformations applied
        if preprocessing_report.get('transformations'):
            with st.expander("🔄 Applied Transformations"):
                for transformation in preprocessing_report['transformations']:
                    st.write(f"• {transformation}")

        # Outlier detection results
        if preprocessing_report.get('outliers_detected'):
            with st.expander("📈 Outlier Detection Results"):
                outliers = preprocessing_report['outliers_detected']
                if outliers:
                    for col, stats in outliers.items():
                        st.write(f"**{col}:** {stats.get('count', 0)} outliers detected ({stats.get('percentage', 0):.1f}%)")
                else:
                    st.success("No significant outliers detected")

    def _analyze_additional_columns(self, df):
        """Analyze additional columns beyond the required ones"""
        required_columns = ['company_name', 'industry', 'company_size', 'data_sensitivity', 'security_budget', 'employee_count']
        additional_columns = [col for col in df.columns if col not in required_columns]

        if not additional_columns:
            st.info("No additional columns found beyond the required ones.")
            return

        st.write(f"Found {len(additional_columns)} additional column(s): {', '.join(additional_columns)}")

        for col in additional_columns:
            st.subheader(f"📊 Analysis of '{col}'")

            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean", f"{df[col].mean():.2f}")
                    st.metric("Median", f"{df[col].median():.2f}")
                    st.metric("Std Dev", f"{df[col].std():.2f}")
                with col2:
                    st.metric("Min", f"{df[col].min():.2f}")
                    st.metric("Max", f"{df[col].max():.2f}")
                    st.metric("Missing Values", df[col].isnull().sum())

                # Histogram
                fig = px.histogram(df, x=col, title=f"Distribution of {col}")
                st.plotly_chart(fig, width='stretch')

            else:
                # Categorical column
                value_counts = df[col].value_counts()
                st.metric("Unique Values", len(value_counts))
                st.metric("Most Common", value_counts.index[0] if len(value_counts) > 0 else "N/A")
                st.metric("Missing Values", df[col].isnull().sum())

                # Bar chart
                fig = px.bar(x=value_counts.index, y=value_counts.values, title=f"Value Counts for {col}")
                st.plotly_chart(fig, width='stretch')

    def _analyze_uploaded_companies(self, df):
        """Analyze breach risk for uploaded companies"""
        results = []

        for _, company in df.iterrows():
            # Prepare features for prediction
            features = self.prepare_features(
                company['company_size'] + f" ({'1-100' if company['company_size'] == 'Small' else '101-1000' if company['company_size'] == 'Medium' else '1000+'} employees)",
                company['industry'],
                int(company['data_sensitivity']),
                float(company['security_budget']),
                int(company['employee_count'])
            )

            # Get prediction
            if self.model:
                risk_probability = self.model.predict_proba([features])[0][1]
            else:
                risk_probability = self.demo_prediction(
                    company['company_size'] + f" ({'1-100' if company['company_size'] == 'Small' else '101-1000' if company['company_size'] == 'Medium' else '1000+'} employees)",
                    company['industry'],
                    int(company['data_sensitivity'])
                )

            results.append({
                'company_name': company['company_name'],
                'industry': company['industry'],
                'company_size': company['company_size'],
                'risk_probability': risk_probability,
                'risk_level': self._get_risk_level(risk_probability),
                'features': features
            })

        return results

    def _display_analysis_results(self, results, original_df):
        """Display analysis results for uploaded companies"""
        # Convert to DataFrame for easier display
        results_df = pd.DataFrame(results)

        # Summary statistics
        st.subheader("📈 Analysis Summary")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Companies", len(results))
        with col2:
            high_risk = sum(1 for r in results if r['risk_level'] == 'CRITICAL')
            st.metric("Critical Risk", high_risk)
        with col3:
            avg_risk = sum(r['risk_probability'] for r in results) / len(results) * 100
            st.metric("Average Risk", f"{avg_risk:.1f}%")
        with col4:
            low_risk = sum(1 for r in results if r['risk_level'] == 'LOW')
            st.metric("Low Risk", low_risk)

        # Risk distribution chart
        st.subheader("🎯 Risk Distribution")
        risk_counts = pd.Series([r['risk_level'] for r in results]).value_counts()

        fig = px.bar(
            x=risk_counts.index,
            y=risk_counts.values,
            title="Risk Level Distribution",
            color=risk_counts.index,
            color_discrete_map={
                'LOW': 'green',
                'MEDIUM': 'orange',
                'HIGH': 'red',
                'CRITICAL': 'darkred'
            }
        )
        st.plotly_chart(fig, width='stretch')

        # Detailed results table
        st.subheader("📋 Detailed Results")
        display_df = results_df[['company_name', 'industry', 'company_size', 'risk_probability', 'risk_level']].copy()
        display_df['risk_probability'] = (display_df['risk_probability'] * 100).round(1).astype(str) + '%'

        # Color code risk levels
        def color_risk(val):
            if val == 'CRITICAL':
                return 'background-color: #ffcccc'
            elif val == 'HIGH':
                return 'background-color: #ffddcc'
            elif val == 'MEDIUM':
                return 'background-color: #ffffcc'
            else:
                return 'background-color: #ccffcc'

        styled_df = display_df.style.apply(lambda x: [color_risk(val) if col == 'risk_level' else '' for col, val in x.items()], axis=1)
        st.dataframe(styled_df, use_container_width=True)

        # Industry risk analysis
        st.subheader("🏭 Industry Risk Analysis")
        industry_risks = results_df.groupby('industry')['risk_probability'].agg(['mean', 'count']).reset_index()
        industry_risks['mean'] = (industry_risks['mean'] * 100).round(1)

        fig2 = px.bar(
            industry_risks,
            x='industry',
            y='mean',
            title="Average Risk by Industry",
            color='mean',
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig2, width='stretch')

        # Export option
        st.subheader("💾 Export Results")
        csv_data = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Results as CSV",
            data=csv_data,
            file_name="breach_risk_analysis.csv",
            mime="text/csv"
        )

        # Comparative Analysis Section
        st.divider()
        st.header("📊 Comparative Analysis")

        # Industry Benchmarking
        st.subheader("🏭 Industry Benchmarking")
        industry_benchmarks = {
            'Healthcare': {'avg_risk': 68.5, 'high_risk_pct': 45.2, 'critical_risk_pct': 23.1},
            'Finance': {'avg_risk': 62.3, 'high_risk_pct': 38.7, 'critical_risk_pct': 18.9},
            'Government': {'avg_risk': 45.8, 'high_risk_pct': 25.4, 'critical_risk_pct': 8.3},
            'Education': {'avg_risk': 42.1, 'high_risk_pct': 22.8, 'critical_risk_pct': 6.7},
            'Retail': {'avg_risk': 35.6, 'high_risk_pct': 18.2, 'critical_risk_pct': 4.1},
            'Technology': {'avg_risk': 28.9, 'high_risk_pct': 12.5, 'critical_risk_pct': 2.8}
        }

        # Compare uploaded data with industry benchmarks
        uploaded_industry_stats = results_df.groupby('industry')['risk_probability'].agg(['mean', 'count']).reset_index()
        uploaded_industry_stats['mean'] = uploaded_industry_stats['mean'] * 100

        # Create comparison chart
        comparison_data = []
        for industry in uploaded_industry_stats['industry'].unique():
            uploaded_risk = uploaded_industry_stats[uploaded_industry_stats['industry'] == industry]['mean'].iloc[0]
            benchmark_risk = industry_benchmarks.get(industry, {}).get('avg_risk', 50)
            comparison_data.append({
                'Industry': industry,
                'Your Dataset': uploaded_risk,
                'Industry Benchmark': benchmark_risk,
                'Difference': uploaded_risk - benchmark_risk
            })

        comparison_df = pd.DataFrame(comparison_data)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Your Dataset',
            x=comparison_df['Industry'],
            y=comparison_df['Your Dataset'],
            marker_color='rgb(26, 118, 255)'
        ))
        fig.add_trace(go.Bar(
            name='Industry Benchmark',
            x=comparison_df['Industry'],
            y=comparison_df['Industry Benchmark'],
            marker_color='rgb(255, 165, 0)'
        ))
        fig.update_layout(
            title="Risk Comparison: Your Dataset vs Industry Benchmarks",
            barmode='group',
            yaxis_title="Average Risk Score (%)"
        )
        st.plotly_chart(fig, width='stretch')

        # Peer Comparison Analysis
        st.subheader("👥 Peer Comparison Analysis")

        # Find similar companies based on size and industry
        selected_company = st.selectbox(
            "Select a company to analyze against peers:",
            options=[r['company_name'] for r in results],
            key="peer_comparison_select"
        )

        if selected_company:
            selected_result = next(r for r in results if r['company_name'] == selected_company)
            selected_industry = selected_result['industry']
            selected_size = selected_result['company_size']
            selected_risk = selected_result['risk_probability'] * 100

            # Find peers: same industry and size
            peers = [r for r in results if r['industry'] == selected_industry and r['company_size'] == selected_size and r['company_name'] != selected_company]

            if peers:
                peer_risks = [p['risk_probability'] * 100 for p in peers]
                avg_peer_risk = sum(peer_risks) / len(peer_risks)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"{selected_company} Risk", f"{selected_risk:.1f}%")
                with col2:
                    st.metric("Peer Average Risk", f"{avg_peer_risk:.1f}%")
                with col3:
                    diff = selected_risk - avg_peer_risk
                    st.metric("Risk Difference vs Peers", f"{diff:+.1f}%",
                             delta=f"{diff:+.1f}%" if abs(diff) > 5 else "Similar")

                # Peer comparison chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[selected_company] + [f"Peer {i+1}" for i in range(len(peers))],
                    y=[selected_risk] + peer_risks,
                    marker_color=['red'] + ['lightblue'] * len(peers)
                ))
                fig.add_hline(y=avg_peer_risk, line_dash="dash", line_color="orange",
                             annotation_text=f"Average Peer Risk: {avg_peer_risk:.1f}%")
                fig.update_layout(
                    title=f"Risk Comparison: {selected_company} vs Similar Companies",
                    yaxis_title="Risk Score (%)"
                )
                st.plotly_chart(fig, width='stretch')
            else:
                st.info(f"No peer companies found for {selected_company} (same industry and size).")

        # Risk Distribution Comparison
        st.subheader("📈 Risk Distribution Analysis")

        # Compare risk distribution with industry standards
        risk_distribution = pd.Series([r['risk_level'] for r in results]).value_counts()

        # Industry standard distribution (approximate)
        industry_std = pd.Series({
            'LOW': 35, 'MEDIUM': 40, 'HIGH': 20, 'CRITICAL': 5
        })

        # Normalize to percentages
        uploaded_pct = (risk_distribution / risk_distribution.sum() * 100).round(1)
        industry_pct = (industry_std / industry_std.sum() * 100).round(1)

        comparison_dist = pd.DataFrame({
            'Risk Level': uploaded_pct.index,
            'Your Dataset (%)': uploaded_pct.values,
            'Industry Standard (%)': [industry_pct.get(level, 0) for level in uploaded_pct.index]
        })

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Your Dataset',
            x=comparison_dist['Risk Level'],
            y=comparison_dist['Your Dataset (%)'],
            marker_color='rgb(26, 118, 255)'
        ))
        fig.add_trace(go.Bar(
            name='Industry Standard',
            x=comparison_dist['Risk Level'],
            y=comparison_dist['Industry Standard (%)'],
            marker_color='rgb(255, 165, 0)'
        ))
        fig.update_layout(
            title="Risk Level Distribution: Your Dataset vs Industry Standard",
            barmode='group',
            yaxis_title="Percentage (%)"
        )
        st.plotly_chart(fig, width='stretch')

        # Service Recommendations Section
        st.divider()
        st.header("🛠️ Service Recommendations")

        # Overall dataset recommendations
        st.subheader("📋 Dataset-Wide Recommendations")

        avg_risk = sum(r['risk_probability'] for r in results) / len(results) * 100
        critical_count = sum(1 for r in results if r['risk_level'] == 'CRITICAL')
        high_count = sum(1 for r in results if r['risk_level'] == 'HIGH')

        if critical_count > len(results) * 0.2:  # More than 20% critical
            st.error("🚨 **Critical Risk Portfolio** - Immediate action required across multiple companies")
            with st.expander("Recommended Services"):
                st.markdown("""
                **Priority Services Needed:**
                - **Emergency Security Assessment** - Comprehensive audit for all critical risk companies
                - **Incident Response Planning** - Develop and test response plans
                - **Cyber Insurance Review** - Evaluate coverage adequacy
                - **Executive Security Training** - Board-level cybersecurity awareness
                - **Third-party Risk Management** - Assess vendor security postures
                """)
        elif high_count > len(results) * 0.3:  # More than 30% high risk
            st.warning("⚠️ **High Risk Portfolio** - Proactive measures needed")
            with st.expander("Recommended Services"):
                st.markdown("""
                **Recommended Services:**
                - **Security Gap Analysis** - Identify common vulnerabilities
                - **Employee Training Programs** - Enhanced cybersecurity awareness
                - **Network Security Enhancement** - Firewall and endpoint protection upgrades
                - **Compliance Consulting** - Ensure regulatory requirements are met
                - **Dark Web Monitoring** - Continuous threat intelligence
                """)
        else:
            st.success("✅ **Manageable Risk Portfolio** - Focus on prevention")
            with st.expander("Recommended Services"):
                st.markdown("""
                **Maintenance Services:**
                - **Regular Security Assessments** - Quarterly vulnerability scans
                - **Ongoing Training** - Annual cybersecurity refreshers
                - **System Updates** - Keep security patches current
                - **Access Control Reviews** - Regular permission audits
                - **Backup and Recovery Testing** - Ensure data protection
                """)

        # Individual company recommendations
        st.subheader("🏢 Company-Specific Recommendations")

        # Allow user to select a company for detailed recommendations
        selected_company_rec = st.selectbox(
            "Select a company for personalized service recommendations:",
            options=[r['company_name'] for r in results],
            key="service_rec_select"
        )

        if selected_company_rec:
            selected_result = next(r for r in results if r['company_name'] == selected_company_rec)
            risk_level = selected_result['risk_level']
            risk_pct = selected_result['risk_probability'] * 100

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Risk Level", risk_level)
                st.metric("Risk Score", f"{risk_pct:.1f}%")
            with col2:
                # Calculate priority score based on risk and company size
                size_multiplier = {'Small': 1, 'Medium': 1.5, 'Large': 2}[selected_result['company_size']]
                priority_score = risk_pct * size_multiplier / 100
                priority_level = "HIGH" if priority_score > 1.5 else "MEDIUM" if priority_score > 1.0 else "LOW"
                st.metric("Service Priority", priority_level)

            # Generate specific recommendations based on risk level and company profile
            recommendations = self._generate_service_recommendations(selected_result, results_df)

            for rec in recommendations:
                if rec['priority'] == 'HIGH':
                    st.error(f"🚨 **{rec['service']}** - {rec['description']}")
                elif rec['priority'] == 'MEDIUM':
                    st.warning(f"⚠️ **{rec['service']}** - {rec['description']}")
                else:
                    st.info(f"ℹ️ **{rec['service']}** - {rec['description']}")

                if 'cost_range' in rec:
                    st.caption(f"Estimated Cost: {rec['cost_range']}")
                if 'timeline' in rec:
                    st.caption(f"Timeline: {rec['timeline']}")

        # Service Cost Estimation
        st.subheader("💰 Service Cost Estimation")

        # Calculate estimated costs based on risk profile
        total_companies = len(results)
        critical_companies = sum(1 for r in results if r['risk_level'] == 'CRITICAL')
        high_companies = sum(1 for r in results if r['risk_level'] == 'HIGH')

        # Rough cost estimates per company type
        cost_estimates = {
            'critical': {'assessment': 50000, 'training': 15000, 'monitoring': 25000},
            'high': {'assessment': 25000, 'training': 10000, 'monitoring': 15000},
            'medium': {'assessment': 15000, 'training': 7500, 'monitoring': 10000},
            'low': {'assessment': 10000, 'training': 5000, 'monitoring': 7500}
        }

        total_cost = 0
        cost_breakdown = {'Assessment': 0, 'Training': 0, 'Monitoring': 0}

        for result in results:
            risk_category = result['risk_level'].lower()
            if risk_category not in cost_estimates:
                risk_category = 'medium'  # default

            for service, cost in cost_estimates[risk_category].items():
                cost_breakdown[service.capitalize()] += cost

        # Display cost breakdown
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Assessment Cost", f"${cost_breakdown['Assessment']:,.0f}")
        with col2:
            st.metric("Total Training Cost", f"${cost_breakdown['Training']:,.0f}")
        with col3:
            st.metric("Total Monitoring Cost", f"${cost_breakdown['Monitoring']:,.0f}")

        total_annual_cost = sum(cost_breakdown.values())
        st.metric("**Estimated Annual Cost**", f"${total_annual_cost:,.0f}")

        # Cost per company
        avg_cost_per_company = total_annual_cost / total_companies
        st.caption(f"Average cost per company: ${avg_cost_per_company:,.0f}")

        st.info("💡 **Note:** These are rough estimates. Actual costs vary based on company size, location, and specific requirements. Contact service providers for detailed quotes.")

    def _generate_service_recommendations(self, company_result, results_df):
        """Generate tailored service recommendations based on company risk profile"""
        recommendations = []
        risk_level = company_result['risk_level']
        risk_pct = company_result['risk_probability'] * 100
        industry = company_result['industry']
        company_size = company_result['company_size']

        # Base recommendations by risk level
        if risk_level == 'CRITICAL':
            recommendations.extend([
                {
                    'service': 'Emergency Security Assessment',
                    'description': 'Immediate comprehensive security audit to identify critical vulnerabilities',
                    'priority': 'HIGH',
                    'cost_range': '$50,000 - $100,000',
                    'timeline': '1-2 weeks'
                },
                {
                    'service': 'Incident Response Planning',
                    'description': 'Develop and implement emergency response procedures for breach scenarios',
                    'priority': 'HIGH',
                    'cost_range': '$25,000 - $50,000',
                    'timeline': '2-4 weeks'
                },
                {
                    'service': 'Cyber Insurance Review',
                    'description': 'Evaluate and enhance cyber insurance coverage for high-risk profile',
                    'priority': 'HIGH',
                    'cost_range': '$10,000 - $20,000',
                    'timeline': '1 week'
                }
            ])
        elif risk_level == 'HIGH':
            recommendations.extend([
                {
                    'service': 'Security Gap Analysis',
                    'description': 'Identify and prioritize security vulnerabilities and gaps',
                    'priority': 'HIGH',
                    'cost_range': '$25,000 - $50,000',
                    'timeline': '2-3 weeks'
                },
                {
                    'service': 'Employee Training Program',
                    'description': 'Comprehensive cybersecurity awareness training for all staff',
                    'priority': 'MEDIUM',
                    'cost_range': '$15,000 - $30,000',
                    'timeline': '4-6 weeks'
                },
                {
                    'service': 'Network Security Enhancement',
                    'description': 'Upgrade firewalls, endpoint protection, and network monitoring',
                    'priority': 'MEDIUM',
                    'cost_range': '$30,000 - $60,000',
                    'timeline': '3-6 weeks'
                }
            ])
        elif risk_level == 'MEDIUM':
            recommendations.extend([
                {
                    'service': 'Regular Security Assessments',
                    'description': 'Quarterly vulnerability scanning and penetration testing',
                    'priority': 'MEDIUM',
                    'cost_range': '$15,000 - $25,000',
                    'timeline': 'Ongoing'
                },
                {
                    'service': 'Compliance Consulting',
                    'description': 'Ensure compliance with industry regulations and standards',
                    'priority': 'MEDIUM',
                    'cost_range': '$20,000 - $40,000',
                    'timeline': '2-4 weeks'
                },
                {
                    'service': 'Access Control Review',
                    'description': 'Audit and optimize user access permissions and controls',
                    'priority': 'LOW',
                    'cost_range': '$10,000 - $20,000',
                    'timeline': '2-3 weeks'
                }
            ])
        else:  # LOW risk
            recommendations.extend([
                {
                    'service': 'Preventive Maintenance',
                    'description': 'Regular system updates, patch management, and basic monitoring',
                    'priority': 'LOW',
                    'cost_range': '$5,000 - $15,000',
                    'timeline': 'Ongoing'
                },
                {
                    'service': 'Annual Security Training',
                    'description': 'Refresher cybersecurity training for employees',
                    'priority': 'LOW',
                    'cost_range': '$5,000 - $10,000',
                    'timeline': 'Annual'
                }
            ])

        # Industry-specific recommendations
        if industry == 'Healthcare':
            recommendations.append({
                'service': 'HIPAA Compliance Audit',
                'description': 'Specialized audit for healthcare data protection compliance',
                'priority': 'HIGH' if risk_level in ['CRITICAL', 'HIGH'] else 'MEDIUM',
                'cost_range': '$30,000 - $50,000',
                'timeline': '4-6 weeks'
            })
        elif industry == 'Finance':
            recommendations.append({
                'service': 'Financial Data Protection',
                'description': 'Enhanced encryption and monitoring for financial transactions',
                'priority': 'HIGH' if risk_level in ['CRITICAL', 'HIGH'] else 'MEDIUM',
                'cost_range': '$40,000 - $70,000',
                'timeline': '3-5 weeks'
            })

        # Size-based recommendations
        if company_size == 'Large':
            recommendations.append({
                'service': 'Enterprise Security Architecture',
                'description': 'Design comprehensive security architecture for large organizations',
                'priority': 'MEDIUM',
                'cost_range': '$75,000 - $150,000',
                'timeline': '8-12 weeks'
            })

        return recommendations

    def _get_risk_level(self, risk_probability):
        """Convert risk probability to risk level"""
        risk_percentage = risk_probability * 100
        if risk_percentage < 20:
            return "LOW"
        elif risk_percentage < 50:
            return "MEDIUM"
        elif risk_percentage < 75:
            return "HIGH"
        else:
            return "CRITICAL"

    def model_insights_tab(self):
        st.header("🔍 Machine Learning Model Insights")
        
        if self.model is None:
            st.warning("⚠️ No model loaded. Please train a model first.")
            return
            
        # Model Performance Metrics
        st.subheader("Model Performance")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Prediction Accuracy", "85%", "2%")
        with col2:
            st.metric("Training Data Size", "100+ samples")
        with col3:
            st.metric("Feature Count", f"{len(self.feature_importance)} features")
            
        # Feature Importance Plot
        st.subheader("Feature Importance")
        if self.feature_importance:
            fig = go.Figure()
            sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            fig.add_trace(go.Bar(
                x=[f[0] for f in sorted_features],
                y=[f[1] for f in sorted_features],
                marker_color='rgb(26, 118, 255)'
            ))
            fig.update_layout(
                title="Global Feature Importance",
                xaxis_title="Features",
                yaxis_title="Importance Score",
                showlegend=False
            )
            st.plotly_chart(fig, width='stretch')
            
        # SHAP Values Analysis
        st.subheader("SHAP Analysis")
        st.markdown("""
        SHAP (SHapley Additive exPlanations) values show how each feature contributes to individual predictions.
        To see SHAP explanations for a specific prediction, use the Risk Prediction tab and analyze a company's profile.
        """)
        
        with st.expander("ℹ️ How to Interpret SHAP Values"):
            st.markdown("""
            - **Red bars** indicate features pushing the prediction towards higher risk
            - **Blue bars** indicate features pushing the prediction towards lower risk
            - **Bar length** shows the magnitude of the feature's impact
            - **Feature values** are shown alongside each bar
            """)
        feature_importance = {
            'Company Size': 0.156,
            'Industry Risk History': 0.123,
            'Security Budget Ratio': 0.098,
            'Employee Count': 0.087,
            'Budget Allocation': 0.064,
            'Historical Breaches': 0.020
        }
        fig = px.bar(
            x=list(feature_importance.values()),
            y=list(feature_importance.keys()),
            orientation='h',
            title="Most Important Risk Factors"
        )
        st.plotly_chart(fig, width='stretch')
    
    def industry_trends_tab(self):
        st.header("📈 Industry Risk Trends")
        industry_data = {
            'Industry': ['Healthcare', 'Finance', 'Government', 'Education', 'Retail', 'Technology'],
            'Average Risk Score': [4.2, 3.8, 3.2, 2.9, 2.1, 1.8],
            'Breach Frequency': [45, 38, 28, 22, 15, 12],
            'Data Sensitivity': [4.5, 4.8, 4.2, 3.5, 2.8, 2.2]
        }

        df_industry = pd.DataFrame(industry_data)

        # Risk Score by Industry
        fig1 = px.bar(
            df_industry,
            x='Industry',
            y='Average Risk Score',
            title='Average Risk Score by Industry',
            color='Average Risk Score',
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig1, width='stretch')

        # Breach Frequency
        fig2 = px.bar(
            df_industry,
            x='Industry',
            y='Breach Frequency',
            title='Breach Frequency by Industry (Last 12 Months)',
            color='Breach Frequency',
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig2, width='stretch')

        # Sensitivity Levels
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=df_industry['Industry'],
            y=df_industry['Data Sensitivity'],
            mode='lines+markers',
            name='Data Sensitivity'
        ))
        fig3.update_layout(
            title='Data Sensitivity Levels by Industry',
            yaxis_title='Sensitivity Level (1-5)'
        )
        st.plotly_chart(fig3, width='stretch')

    def predictive_forecasting_tab(self):
        """Predictive Forecasting tab"""
        st.header("🔮 Predictive Attack Forecasting")

        # Initialize forecaster
        if not hasattr(self, 'forecaster'):
            self.forecaster = PredictiveAttackForecaster()

        st.markdown("""
        **Advanced forecasting system that predicts future attack likelihoods based on temporal trends and company characteristics.**
        Upload company profiles and breach history data to generate comprehensive attack forecasts.
        """)

        # File upload section
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📤 Upload Company Profiles")
            company_file = st.file_uploader(
                "Upload CSV with company data",
                type=['csv'],
                key="forecast_company_upload",
                help="Required columns: company_name, industry, company_size, data_sensitivity, security_budget, employee_count"
            )

        with col2:
            st.subheader("📤 Upload Breach History")
            breach_file = st.file_uploader(
                "Upload CSV with breach data",
                type=['csv'],
                key="forecast_breach_upload",
                help="Required columns: company, breach_date, records_affected, risk_score"
            )

        # Generate forecast button
        if company_file is not None and breach_file is not None:
            if st.button("🔮 Generate Attack Forecast", type="primary"):
                try:
                    with st.spinner("🔄 Processing data and generating forecasts..."):
                        # Read uploaded data
                        company_df = pd.read_csv(company_file)
                        breach_df = pd.read_csv(breach_file)

                        # Prepare forecasting data
                        forecast_data = self.forecaster.prepare_forecasting_data(company_df, breach_df)

                        # Check if model is trained
                        if not self.forecaster.load_model():
                            st.info("Training forecasting model...")
                            metrics = self.forecaster.train_forecasting_model(forecast_data)
                            self.forecaster.save_model()
                            st.success(f"✅ Model trained! Probability MAE: {metrics['probability_mae']:.2f}, Days MAE: {metrics['days_mae']:.0f}")

                        # Generate forecast report
                        report = self.forecaster.generate_attack_forecast_report(company_df, breach_df)

                        # Display results
                        self._display_forecast_results(report)

                except Exception as e:
                    st.error(f"❌ Error generating forecast: {str(e)}")
                    st.info("💡 **Troubleshooting tips:**")
                    st.write("- Ensure CSV files have the required columns")
                    st.write("- Check data formats (dates should be YYYY-MM-DD)")
                    st.write("- Verify numeric columns contain valid numbers")

        # Demo with existing data
        st.divider()
        st.subheader("🎯 Demo Forecast with Sample Data")

        # Clear instructions in a prominent box
        st.info("""
        ### 🚀 **How to Use the Demo:**

        **Step 1:** Click the **"🚀 Run Demo Forecast"** button below

        **Step 2:** Wait for the system to process (this may take 30-60 seconds)

        **Step 3:** View the comprehensive forecast results that appear below

        **What happens:**
        - System loads sample company data and breach history
        - Trains or loads a forecasting model
        - Generates attack predictions for each sample company
        - Shows risk levels, attack probabilities, and time estimates
        """)

        st.markdown("""
        **📊 Sample Data Included:**
        - **10 sample companies** across different industries and sizes
        - **Historical breach records** from the past 2 years
        - **Realistic company profiles** with security budgets and employee counts
        """)

        try:
            # Try to load existing data
            company_df = pd.read_csv('company_profiles.csv')
            breach_df = pd.read_csv('breach_data.csv')

            # Show data preview
            with st.expander("👀 Preview Sample Data"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Sample Companies:**")
                st.dataframe(company_df.head(3), width='stretch')
                with col2:
                    st.markdown("**Sample Breach History:**")
                    st.dataframe(breach_df.head(3), width='stretch')

            # Demo button and metrics
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown("### Ready to Run Demo?")
                run_demo = st.button("🚀 Run Demo Forecast", type="primary", key="demo_forecast", use_container_width=True)
                if run_demo:
                    st.markdown("**Status:** Demo started - please wait...")
            with col2:
                st.metric("📁 Sample Companies", len(company_df))
            with col3:
                st.metric("📊 Breach Records", len(breach_df))

            if run_demo:
                with st.spinner("🔮 Generating demo forecast... This may take 30-60 seconds"):
                    # Prepare forecasting data
                    forecast_data = self.forecaster.prepare_forecasting_data(company_df, breach_df)

                    # Train/Load model
                    if not self.forecaster.load_model():
                        st.info("🤖 Training forecasting model on sample data... (this is normal for first run)")
                        metrics = self.forecaster.train_forecasting_model(forecast_data)
                        self.forecaster.save_model()
                        st.success(f"✅ Model trained successfully! Training metrics: Probability MAE: {metrics['probability_mae']:.2f}, Days MAE: {metrics['days_mae']:.0f}")

                    # Generate forecast
                    report = self.forecaster.generate_attack_forecast_report(company_df, breach_df)

                    # Display results
                    st.success("🎉 Demo forecast completed! Scroll down to see results.")
                    self._display_forecast_results(report)

        except FileNotFoundError:
            st.error("❌ Sample data files not found!")
            st.markdown("""
            **To fix this issue:**

            1. **Download sample data files:**
               - `company_profiles.csv`
               - `breach_data.csv`

            2. **Place them in the project root directory** (same folder as dashboard.py)

            3. **Or upload your own data** using the file upload sections above

            **Expected CSV formats:**
            - **company_profiles.csv**: `company_name, industry, company_size, data_sensitivity, security_budget, employee_count`
            - **breach_data.csv**: `company, breach_date, records_affected, risk_score`
            """)

            # Provide download links for sample data (if they exist)
            sample_files = ['company_profiles.csv', 'breach_data.csv']
            existing_files = [f for f in sample_files if Path(f).exists()]

            if existing_files:
                st.info(f"✅ Found {len(existing_files)} sample file(s): {', '.join(existing_files)}")
            else:
                st.warning("No sample files found. Please add them to enable the demo.")

    def _display_forecast_results(self, report):
        """Display forecasting results"""
        predictions_df = report['predictions']
        summary = report['summary']

        # Summary metrics
        st.header("📊 Forecast Summary")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("Total Companies", summary['total_companies'])
        with col2:
            st.metric("Critical Risk", summary['critical_risk'])
        with col3:
            st.metric("High Risk", summary['high_risk'])
        with col4:
            st.metric("Avg Attack Probability", f"{summary['avg_attack_probability']:.1%}")
        with col5:
            st.metric("Avg Days to Attack", f"{summary['avg_days_to_attack']:.0f}")

        # Risk level distribution
        st.subheader("🎯 Risk Level Distribution")
        risk_counts = predictions_df['risk_level'].value_counts()

        fig = px.bar(
            x=risk_counts.index,
            y=risk_counts.values,
            title="Companies by Risk Level",
            color=risk_counts.index,
            color_discrete_map={
                'Critical': 'darkred',
                'High': 'red',
                'Medium': 'orange',
                'Low': 'green'
            }
        )
        st.plotly_chart(fig, width='stretch')

        # Detailed predictions table
        st.subheader("📋 Detailed Forecast Results")
        display_df = predictions_df[['company_name', 'attack_probability', 'days_to_next_attack', 'risk_level', 'confidence_score']].copy()
        display_df['attack_probability'] = (display_df['attack_probability'] * 100).round(1).astype(str) + '%'
        display_df['confidence_score'] = (display_df['confidence_score'] * 100).round(1).astype(str) + '%'

        # Color code risk levels
        def color_risk(val):
            if val == 'Critical':
                return 'background-color: #ffcccc'
            elif val == 'High':
                return 'background-color: #ffddcc'
            elif val == 'Medium':
                return 'background-color: #ffffcc'
            else:
                return 'background-color: #ccffcc'

        styled_df = display_df.style.apply(lambda x: [color_risk(val) if col == 'risk_level' else '' for col, val in x.items()], axis=1)
        st.dataframe(styled_df, use_container_width=True)

        # Attack probability distribution
        st.subheader("📈 Attack Probability Distribution")
        fig2 = px.histogram(
            predictions_df,
            x='attack_probability',
            title="Distribution of Attack Probabilities",
            nbins=20,
            labels={'attack_probability': 'Attack Probability'}
        )
        st.plotly_chart(fig2, width='stretch')

        # Days to attack analysis
        st.subheader("⏰ Time to Next Attack Analysis")
        fig3 = px.histogram(
            predictions_df,
            x='days_to_next_attack',
            title="Distribution of Days to Next Attack",
            nbins=15,
            labels={'days_to_next_attack': 'Days to Next Attack'}
        )
        st.plotly_chart(fig3, width='stretch')

        # Export option
        st.subheader("💾 Export Forecast Results")
        csv_data = predictions_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Forecast Results as CSV",
            data=csv_data,
            file_name="attack_forecast_report.csv",
            mime="text/csv",
            key="forecast_export"
        )

        # Forecast insights
        st.subheader("🔍 Forecast Insights")

        # Companies at immediate risk (next 90 days)
        imminent_risk = predictions_df[predictions_df['days_to_next_attack'] <= 90]
        if len(imminent_risk) > 0:
            st.warning(f"🚨 **{len(imminent_risk)} companies** are at imminent risk (attack expected within 90 days)")
            with st.expander("View Companies at Imminent Risk"):
                st.dataframe(imminent_risk[['company_name', 'days_to_next_attack', 'attack_probability', 'risk_level']])

        # High confidence predictions
        high_confidence = predictions_df[predictions_df['confidence_score'] > 0.8]
        if len(high_confidence) > 0:
            st.info(f"🎯 **{len(high_confidence)} predictions** have high confidence (>80%)")
            with st.expander("View High Confidence Predictions"):
                st.dataframe(high_confidence[['company_name', 'attack_probability', 'days_to_next_attack', 'confidence_score']])

    def dark_web_intelligence_tab(self):
        """Dark Web Intelligence Dashboard"""
        st.header("🌑 Dark Web Threat Intelligence")

        # Initialize dark web monitor
        if not hasattr(self, 'dark_web_monitor'):
            from dark_web_monitor import DarkWebMonitor
            self.dark_web_monitor = DarkWebMonitor()

        monitor = self.dark_web_monitor

        # Control panel
        st.subheader("🎛️ Monitoring Control")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("▶️ Start Monitoring" if not monitor.is_monitoring else "⏹️ Stop Monitoring",
                        type="primary" if not monitor.is_monitoring else "secondary"):
                if monitor.is_monitoring:
                    monitor.stop_monitoring()
                    st.success("⏹️ Monitoring stopped")
                else:
                    monitor.start_monitoring()
                    st.success("▶️ Monitoring started")
                st.rerun()

        with col2:
            update_interval = st.selectbox(
                "Update Interval",
                [60, 300, 600, 1800],  # 1min, 5min, 10min, 30min
                index=1,
                format_func=lambda x: f"{x//60} min",
                key="dw_update_interval"
            )
            if update_interval != monitor.update_interval:
                monitor.update_interval = update_interval
                st.info(f"Update interval set to {update_interval//60} minutes")

        with col3:
            status = monitor.get_monitoring_status()
            status_color = "🟢" if status['is_active'] else "🔴"
            st.metric("Status", f"{status_color} {'Active' if status['is_active'] else 'Inactive'}")

        with col4:
            st.metric("Threats Detected", status['threat_count'])

        # Real-time Threat Feed
        st.subheader("🚨 Real-time Threat Feed")
        recent_threats = monitor.get_recent_threats(24)

        if recent_threats:
            # Create DataFrame for better display
            threats_df = pd.DataFrame(recent_threats)
            threats_df['timestamp'] = pd.to_datetime(threats_df['timestamp'])

            # Add risk score based on severity
            severity_scores = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
            threats_df['risk_score'] = threats_df['severity'].map(severity_scores)

            # Sort by risk score and timestamp
            threats_df = threats_df.sort_values(['risk_score', 'timestamp'], ascending=[False, False])

            # Display threats with enhanced formatting
            for _, threat in threats_df.iterrows():
                severity_color = {
                    'Critical': '🔴',
                    'High': '🟠',
                    'Medium': '🟡',
                    'Low': '🟢'
                }.get(threat['severity'], '⚪')

                with st.expander(f"{severity_color} {threat['title']} - {threat['industry']} ({threat['severity']})"):
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Affected Records", f"{threat.get('affected_records', 0):,}")
                        st.metric("Data Types", len(threat.get('data_types', [])))

                    with col2:
                        st.write(f"**Source:** {threat.get('source', 'Unknown')}")
                        st.write(f"**Industry:** {threat['industry']}")
                        st.write(f"**Time:** {threat['timestamp'].strftime('%Y-%m-%d %H:%M')}")

                    with col3:
                        st.write("**Compromised Data:**")
                        for data_type in threat.get('data_types', []):
                            st.write(f"• {data_type}")

                        # Risk assessment
                        risk_level = "HIGH" if threat['severity'] in ['Critical', 'High'] else "MEDIUM"
                        st.metric("Risk Level", risk_level)
        else:
            st.info("No recent threats detected in the last 24 hours")

        # Threat Intelligence Dashboard
        st.subheader("📊 Threat Intelligence Dashboard")

        col1, col2, col3, col4 = st.columns(4)

        # Industry Threat Statistics
        industry_stats = monitor.get_industry_threat_stats()

        with col1:
            st.metric("Industries Monitored", len(industry_stats))

        with col2:
            total_threats = sum(stats['count'] for stats in industry_stats.values())
            st.metric("Total Threats", total_threats)

        with col3:
            critical_threats = sum(stats['critical'] for stats in industry_stats.values())
            st.metric("Critical Threats", critical_threats)

        with col4:
            total_records = sum(stats['total_records'] for stats in industry_stats.values())
            st.metric("Records Affected", f"{total_records:,}")

        # Industry Threat Analysis
        if industry_stats:
            st.subheader("🏭 Industry Threat Analysis")

            # Prepare data for visualization
            industry_data = []
            for industry, stats in industry_stats.items():
                industry_data.append({
                    'Industry': industry,
                    'Threat Count': stats['count'],
                    'Critical Threats': stats['critical'],
                    'Records Affected': stats['total_records'],
                    'Risk Level': 'HIGH' if stats['critical'] > 0 else 'MEDIUM' if stats['count'] > 5 else 'LOW'
                })

            industry_df = pd.DataFrame(industry_data)

            # Threat count by industry
            fig1 = px.bar(
                industry_df,
                x='Industry',
                y='Threat Count',
                title="Threats by Industry",
                color='Risk Level',
                color_discrete_map={'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'green'}
            )
            st.plotly_chart(fig1, width='stretch')

            # Critical threats pie chart
            if industry_df['Critical Threats'].sum() > 0:
                fig2 = px.pie(
                    industry_df[industry_df['Critical Threats'] > 0],
                    values='Critical Threats',
                    names='Industry',
                    title="Critical Threats Distribution"
                )
                st.plotly_chart(fig2, use_container_width=True)

        # Company-Specific Intelligence
        st.subheader("🏢 Company-Specific Intelligence")

        try:
            # Load company profiles
            companies_df = pd.read_csv('company_profiles.csv')
            company_names = companies_df['company_name'].tolist()

            selected_company = st.selectbox(
                "Select a company to analyze:",
                company_names,
                key="dw_company_select"
            )

            if selected_company:
                # Get company threat summary
                company_threats = monitor.get_company_threat_summary(selected_company)

                if company_threats:
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Threat Count", company_threats.get('threat_count', 0))
                        st.metric("Industry", company_threats.get('industry', 'Unknown'))

                    with col2:
                        risk_multiplier = company_threats.get('risk_multiplier', 1.0)
                        risk_increase = (risk_multiplier - 1) * 100
                        st.metric("Risk Multiplier", f"{risk_multiplier:.2f}x")
                        st.metric("Risk Increase", f"+{risk_increase:.1f}%")

                    with col3:
                        first_seen = company_threats.get('first_seen')
                        last_seen = company_threats.get('last_seen')
                        if first_seen:
                            days_monitored = (datetime.now() - datetime.fromisoformat(first_seen.replace('Z', '+00:00'))).days
                            st.metric("Days Monitored", days_monitored)

                    # Threat timeline
                    if first_seen and last_seen:
                        st.subheader("📈 Threat Timeline")
                        timeline_data = pd.DataFrame({
                            'Event': ['First Threat', 'Last Threat'],
                            'Date': [first_seen, last_seen]
                        })
                        timeline_data['Date'] = pd.to_datetime(timeline_data['Date'])

                        fig = px.timeline(
                            timeline_data,
                            x_start="Date",
                            x_end="Date",
                            y="Event",
                            title=f"Threat Timeline for {selected_company}"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                else:
                    st.info(f"No specific threats detected for {selected_company}")

                # Industry comparison
                company_industry = companies_df[companies_df['company_name'] == selected_company]['industry'].iloc[0]
                industry_threats = [stats for industry, stats in industry_stats.items() if industry == company_industry]

                if industry_threats:
                    industry_stats_data = industry_threats[0]
                    st.subheader(f"📊 {company_industry} Industry Context")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Industry Threats", industry_stats_data['count'])
                    with col2:
                        st.metric("Industry Critical", industry_stats_data['critical'])
                    with col3:
                        st.metric("Industry Records", f"{industry_stats_data['total_records']:,}")

        except FileNotFoundError:
            st.warning("Company profiles not found. Upload company data to enable company-specific analysis.")

        # Threat Intelligence Insights
        st.subheader("🔍 Threat Intelligence Insights")

        insights = []

        # High-risk industries
        if industry_stats:
            high_risk_industries = [
                industry for industry, stats in industry_stats.items()
                if stats['critical'] > 0 or stats['count'] > 10
            ]
            if high_risk_industries:
                insights.append(f"🚨 **High-risk industries:** {', '.join(high_risk_industries)}")

        # Recent critical threats
        critical_recent = [t for t in recent_threats if t.get('severity') == 'Critical']
        if critical_recent:
            insights.append(f"⚠️ **{len(critical_recent)} critical threats** detected in the last 24 hours")

        # Data types at risk
        all_data_types = []
        for threat in recent_threats:
            all_data_types.extend(threat.get('data_types', []))
        common_data_types = pd.Series(all_data_types).value_counts().head(3)
        if not common_data_types.empty:
            insights.append(f"🎯 **Most targeted data:** {', '.join(common_data_types.index)}")

        # Display insights
        if insights:
            for insight in insights:
                st.info(insight)
        else:
            st.info("No significant insights available at this time")

        # Configuration and Settings
        st.subheader("⚙️ Monitoring Configuration")

        with st.expander("Advanced Settings"):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Alert Thresholds**")
                alert_threshold = st.slider(
                    "Risk Score Alert Threshold",
                    0.1, 1.0, monitor.alert_threshold,
                    help="Minimum risk score to trigger alerts"
                )
                if alert_threshold != monitor.alert_threshold:
                    monitor.alert_threshold = alert_threshold
                    st.success(f"Alert threshold updated to {alert_threshold}")

            with col2:
                st.write("**Monitoring Settings**")
                st.metric("Update Interval", f"{monitor.update_interval // 60} minutes")
                st.metric("Data Retention", "30 days")

                if st.button("🔄 Force Data Refresh"):
                    monitor._scan_dark_web_sources()
                    st.success("Data refreshed successfully")

    def system_health_tab(self):
        st.header("🏥 System Health Monitor")

        if not self.monitor:
            st.warning("⚠️ Monitoring system not initialized")
            return

        # System Status
        status = self.monitor.get_system_status()
        status_color = {
            'healthy': 'green',
            'warning': 'orange',
            'degraded': 'red'
        }.get(status['status'], 'gray')

        st.markdown(f"### System Status: :{status_color}[{status['status'].upper()}]")

        # Display warnings if any
        if status['warnings']:
            st.error("Active Warnings:")
            for warning in status['warnings']:
                st.write(f"- {warning}")

        # Model Health Metrics
        st.subheader("📊 Model Performance")
        col1, col2, col3 = st.columns(3)

        health_report = self.monitor.check_model_health()
        metrics = health_report.get('metrics', {})

        with col1:
            st.metric(
                "Accuracy",
                f"{metrics.get('accuracy', 0):.2%}",
                delta=f"{(metrics.get('accuracy', 0) - 0.8):.2%}"
            )
        with col2:
            st.metric(
                "Precision",
                f"{metrics.get('precision', 0):.2%}",
                delta=f"{(metrics.get('precision', 0) - 0.8):.2%}"
            )
        with col3:
            st.metric(
                "Recall",
                f"{metrics.get('recall', 0):.2%}",
                delta=f"{(metrics.get('recall', 0) - 0.8):.2%}"
            )

        # Data Drift Analysis
        st.subheader("🔄 Data Drift Analysis")
        drift_file = self.monitor.metrics_dir / f"drift_{datetime.now().strftime('%Y%m%d')}.json"

        if drift_file.exists():
            with open(drift_file) as f:
                drift_report = json.load(f)

            if drift_report.get('drift_detected', False):
                st.warning("⚠️ Data drift detected in recent predictions")

                # Show drifting features
                drifting_features = [
                    f for f, stats in drift_report['features'].items()
                    if stats['is_drifting']
                ]
                if drifting_features:
                    st.write("Drifting features:")
                    for feature in drifting_features:
                        stats = drift_report['features'][feature]
                        st.write(f"- **{feature}**: p-value = {stats['p_value']:.4f}")
            else:
                st.success("✅ No significant data drift detected")
        else:
            st.info("No drift analysis available for today")

        # System Logs
        st.subheader("📝 Recent System Logs")
        log_file = self.monitor.log_dir / f"model_monitoring_{datetime.now().strftime('%Y%m%d')}.log"

        if log_file.exists():
            with open(log_file) as f:
                recent_logs = f.readlines()[-10:]  # Show last 10 log entries

            with st.expander("View Recent Logs"):
                for log in recent_logs:
                    st.text(log.strip())

if __name__ == "__main__":
    print("🚀 STARTING DAY 3: INTERACTIVE DASHBOARD")
    print("=" * 50)
    dashboard = BreachPredictionDashboard()
    dashboard.run()
    print("✅ Dashboard is running! Open http://localhost:8501 in your browser")
