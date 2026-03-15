# Smart Factory App

Smart Factory App is an Industry 4.0 dashboard prototype for predictive maintenance, OEE monitoring, and production analytics. It blends machine health data, operational KPIs, and lightweight machine learning into a single manufacturing view.

## What it does

- Simulates live machine data for temperature, vibration, current, pressure, RPM, and energy.
- Estimates failure risk with a Random Forest model.
- Tracks operational metrics such as OEE, cycle time, and shift performance.
- Presents the data in a dashboard-oriented layout designed for manufacturing use cases.

## Stack

- Python
- Flask
- Pandas
- NumPy
- scikit-learn
- HTML, CSS, JavaScript

## Local setup

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python main.py
```

The app runs on `http://127.0.0.1:8080`.

## Portfolio note

This repository is a portfolio prototype that shows how predictive maintenance concepts can be turned into an operator-friendly web dashboard without heavyweight infrastructure.

## License

MIT
