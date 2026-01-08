# SmartFactory: AI-Driven Predictive Maintenance & MES System

## 1. Executive Summary

This project is a comprehensive Industry 4.0 solution that demonstrates the convergence of Operational Technology (OT) data with Information Technology (IT) analytics. The system provides a unified dashboard for real-time monitoring of machine health, manufacturing execution system (MES) metrics, financial cost analysis, and AI-based failure prediction. It serves as a digital twin prototype, offering a holistic view of factory floor operations.

## 2. Key Features

- **Dashboard Architecture:** A modern "Bento Grid" layout with fixed sidebar navigation provides a professional, high-density SaaS user experience. The interface is fully responsive and designed for at-a-glance monitoring.
- **AI & Analytics:**
    - **Anomaly Detection:** A Random Forest Classifier, trained using Scikit-learn, analyzes a live stream of sensor data to compute a real-time failure risk score.
    - **RUL (Remaining Useful Life):** A predictive estimation of machine longevity, presented in operational hours, enables proactive maintenance scheduling.
- **Financial Metrics:** Real-time energy cost calculation (presented in $/hr) is derived from live power consumption data, providing immediate insight into operational expenditure.
- **Production Tracking (MES):**
    - **OEE (Overall Equipment Effectiveness):** Live monitoring of the primary manufacturing productivity KPI.
    - **Cycle Time Analysis:** Tracks the time required for each production cycle.
    - **Shift Tracking:** Monitors production data segmented by operational shifts.
- **Visualization:**
    - A high-performance Chart.js implementation renders real-time sensor data for temperature and vibration.
    - Line charts use smoothed cubic interpolation (`tension: 0.4`) for clear, readable data representation.

## 3. Technical Stack

- **Backend:** Python 3.11, Flask (Microservice), NumPy, Pandas
- **Frontend:** HTML5, CSS3 (Custom Grid Layout), JavaScript (ES6), Chart.js, Bootstrap 5
- **Machine Learning:** Scikit-learn (RandomForestClassifier)

## 4. Installation and Usage

Follow these steps to set up and run the application locally.

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd <repository-directory>
```

### Step 2: Set Up Environment and Install Dependencies
A virtual environment is recommended.
```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install the required packages
pip install -r requirements.txt
```

### Step 3: Run the Application
```bash
python main.py
```
The application will be running and accessible at `http://127.0.0.1:8080`.
