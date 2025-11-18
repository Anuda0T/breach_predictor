import requests
import json
import time
import threading
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Optional
import streamlit as st
import plotly.express as px

class DarkWebMonitor:
    """
    Real-time dark web monitoring system for breach intelligence
    """

    def __init__(self, update_interval: int = 300):  # 5 minutes default
        self.update_interval = update_interval
        self.is_monitoring = False
        self.monitor_thread = None
        self.last_update = None
        self.threat_data = []
        self.company_mentions = {}
        self.alert_threshold = 0.8  # Risk score threshold for alerts

        # Setup logging
        self.logger = logging.getLogger('dark_web_monitor')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # Data storage
        self.data_dir = Path('data')
        self.data_dir.mkdir(exist_ok=True)
        self.threats_file = self.data_dir / 'dark_web_threats.json'

        # Load existing data
        self.load_threat_data()

    def start_monitoring(self):
        """Start the real-time monitoring thread"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            self.logger.info("Dark web monitoring started")

    def stop_monitoring(self):
        """Stop the monitoring thread"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("Dark web monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                self._scan_dark_web_sources()
                self._update_company_risks()
                self._check_alerts()
                self.last_update = datetime.now()
                self._save_threat_data()
            except Exception as e:
                self.logger.error(f"Monitoring error: {str(e)}")

            time.sleep(self.update_interval)

    def _scan_dark_web_sources(self):
        """Scan various dark web sources for breach data"""
        # Simulate dark web scanning (in real implementation, this would use TOR, APIs, etc.)
        new_threats = self._simulate_dark_web_scan()

        # Add new threats
        for threat in new_threats:
            if threat not in self.threat_data:
                self.threat_data.append(threat)
                self.logger.info(f"New threat detected: {threat.get('title', 'Unknown')}")

        # Keep only recent threats (last 30 days)
        cutoff_date = datetime.now() - timedelta(days=30)
        self.threat_data = [
            t for t in self.threat_data
            if datetime.fromisoformat(t['timestamp']) > cutoff_date
        ]

    def _simulate_dark_web_scan(self) -> List[Dict]:
        """Simulate scanning dark web sources (replace with real implementation)"""
        # This is a simulation - in production, you'd integrate with:
        # - TOR network access
        # - Dark web APIs
        # - Scraping tools
        # - Threat intelligence feeds

        industries = ['Technology', 'Finance', 'Healthcare', 'Retail', 'Government']
        companies = ['TechCorp', 'FinanceInc', 'HealthSys', 'RetailChain', 'GovAgency']

        new_threats = []
        import random

        # Randomly generate some threats (for demo purposes)
        if random.random() < 0.3:  # 30% chance of finding new threats
            threat = {
                'id': f"threat_{int(time.time())}_{random.randint(1000,9999)}",
                'title': f"Data Breach in {random.choice(companies)}",
                'description': f"Compromised credentials and sensitive data from {random.choice(industries)} sector",
                'industry': random.choice(industries),
                'severity': random.choice(['Low', 'Medium', 'High', 'Critical']),
                'timestamp': datetime.now().isoformat(),
                'source': 'Dark Web Forum',
                'data_types': random.sample(['emails', 'passwords', 'credit_cards', 'personal_info'], random.randint(1,4)),
                'affected_records': random.randint(1000, 1000000)
            }
            new_threats.append(threat)

        return new_threats

    def _update_company_risks(self):
        """Update risk scores based on new threat intelligence"""
        # Load company profiles
        try:
            companies_df = pd.read_csv('company_profiles.csv')
        except FileNotFoundError:
            self.logger.warning("Company profiles not found")
            return

        # Update risk scores based on recent threats
        for _, company in companies_df.iterrows():
            company_name = company.get('company_name', f"Company_{company['company_id']}")
            industry = company['industry']

            # Calculate risk increase based on industry threats
            industry_threats = [
                t for t in self.threat_data
                if t['industry'] == industry and
                (datetime.now() - datetime.fromisoformat(t['timestamp'])).days <= 7
            ]

            # Risk multiplier based on recent threats
            risk_multiplier = 1.0 + (len(industry_threats) * 0.1)

            # Update company mentions
            if company_name in self.company_mentions:
                self.company_mentions[company_name]['last_seen'] = datetime.now().isoformat()
                self.company_mentions[company_name]['threat_count'] += len(industry_threats)
            else:
                self.company_mentions[company_name] = {
                    'industry': industry,
                    'first_seen': datetime.now().isoformat(),
                    'last_seen': datetime.now().isoformat(),
                    'threat_count': len(industry_threats),
                    'risk_multiplier': risk_multiplier
                }

    def _check_alerts(self):
        """Check for alerts based on threat intelligence"""
        from alert_system import AlertSystem

        alert_system = AlertSystem()

        for company, data in self.company_mentions.items():
            if data['threat_count'] > 0:
                # Send alert for companies with recent threats
                message = f"🚨 Dark Web Alert: {data['threat_count']} recent threats detected for {company} in {data['industry']} sector"
                alert_system.send_alert(message, 'dark_web_threat')

    def get_recent_threats(self, hours: int = 24) -> List[Dict]:
        """Get threats from the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            t for t in self.threat_data
            if datetime.fromisoformat(t['timestamp']) > cutoff_time
        ]

    def get_company_threat_summary(self, company_name: str) -> Optional[Dict]:
        """Get threat summary for a specific company"""
        return self.company_mentions.get(company_name)

    def get_industry_threat_stats(self) -> Dict:
        """Get threat statistics by industry"""
        stats = {}
        for threat in self.threat_data:
            industry = threat['industry']
            if industry not in stats:
                stats[industry] = {'count': 0, 'critical': 0, 'total_records': 0}

            stats[industry]['count'] += 1
            if threat['severity'] == 'Critical':
                stats[industry]['critical'] += 1
            stats[industry]['total_records'] += threat.get('affected_records', 0)

        return stats

    def load_threat_data(self):
        """Load threat data from file"""
        try:
            if self.threats_file.exists():
                with open(self.threats_file, 'r') as f:
                    data = json.load(f)
                    self.threat_data = data.get('threats', [])
                    self.company_mentions = data.get('mentions', {})
                    self.last_update = data.get('last_update')
                    if self.last_update:
                        self.last_update = datetime.fromisoformat(self.last_update)
        except Exception as e:
            self.logger.error(f"Error loading threat data: {str(e)}")

    def _save_threat_data(self):
        """Save threat data to file"""
        try:
            data = {
                'threats': self.threat_data,
                'mentions': self.company_mentions,
                'last_update': self.last_update.isoformat() if self.last_update else None
            }
            with open(self.threats_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving threat data: {str(e)}")

    def get_monitoring_status(self) -> Dict:
        """Get current monitoring status"""
        return {
            'is_active': self.is_monitoring,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'threat_count': len(self.threat_data),
            'monitored_companies': len(self.company_mentions),
            'update_interval': self.update_interval
        }

# Streamlit component for real-time monitoring
def create_monitoring_dashboard():
    """Create a Streamlit dashboard for dark web monitoring"""
    st.header("🌑 Dark Web Threat Monitor")

    # Initialize monitor
    if 'dark_web_monitor' not in st.session_state:
        st.session_state.dark_web_monitor = DarkWebMonitor()

    monitor = st.session_state.dark_web_monitor

    # Control panel
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("▶️ Start Monitoring" if not monitor.is_monitoring else "⏹️ Stop Monitoring"):
            if monitor.is_monitoring:
                monitor.stop_monitoring()
            else:
                monitor.start_monitoring()
            st.rerun()

    with col2:
        update_interval = st.selectbox(
            "Update Interval",
            [60, 300, 600, 1800],  # 1min, 5min, 10min, 30min
            index=1,
            format_func=lambda x: f"{x//60} min"
        )
        if update_interval != monitor.update_interval:
            monitor.update_interval = update_interval

    with col3:
        status = monitor.get_monitoring_status()
        status_color = "🟢" if status['is_active'] else "🔴"
        st.metric("Status", f"{status_color} {'Active' if status['is_active'] else 'Inactive'}")

    # Status display
    st.subheader("📊 Monitoring Status")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Threats", status['threat_count'])
    with col2:
        st.metric("Monitored Companies", status['monitored_companies'])
    with col3:
        last_update = status['last_update']
        if last_update:
            st.metric("Last Update", datetime.fromisoformat(last_update).strftime("%H:%M:%S"))
        else:
            st.metric("Last Update", "Never")
    with col4:
        st.metric("Update Interval", f"{status['update_interval']//60} min")

    # Recent threats
    st.subheader("🚨 Recent Threats (24h)")
    recent_threats = monitor.get_recent_threats(24)

    if recent_threats:
        threats_df = pd.DataFrame(recent_threats)
        threats_df['timestamp'] = pd.to_datetime(threats_df['timestamp'])
        threats_df = threats_df.sort_values('timestamp', ascending=False)

        # Display threats table
        st.dataframe(
            threats_df[['timestamp', 'title', 'industry', 'severity', 'source', 'affected_records']],
            use_container_width=True,
            column_config={
                'timestamp': st.column_config.DatetimeColumn("Time", format="HH:mm:ss"),
                'affected_records': st.column_config.NumberColumn("Records", format="%d")
            }
        )

        # Threat severity distribution
        severity_counts = threats_df['severity'].value_counts()
        st.bar_chart(severity_counts)
    else:
        st.info("No recent threats detected")

    # Industry threat statistics
    st.subheader("🏭 Industry Threat Statistics")
    industry_stats = monitor.get_industry_threat_stats()

    if industry_stats:
        stats_data = []
        for industry, stats in industry_stats.items():
            stats_data.append({
                'Industry': industry,
                'Threat Count': stats['count'],
                'Critical Threats': stats['critical'],
                'Total Records': stats['total_records']
            })

        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)

        # Visualization
        fig = px.bar(
            stats_df,
            x='Industry',
            y='Threat Count',
            title="Threats by Industry",
            color='Critical Threats'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No industry statistics available")

if __name__ == "__main__":
    # Test the monitoring system
    monitor = DarkWebMonitor(update_interval=60)  # 1 minute for testing
    monitor.start_monitoring()

    print("Dark web monitoring started. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(10)
            status = monitor.get_monitoring_status()
            print(f"Status: {status}")
    except KeyboardInterrupt:
        monitor.stop_monitoring()
        print("Monitoring stopped.")
