# Payor Benchmarking UI

A Streamlit-based web application for analyzing and benchmarking payor data.

## Description

This application provides a user interface for analyzing payor benchmarking data, including:
- Data upload capabilities
- Ranking analysis
- Trend visualization
- Interactive dashboards

## Setup

1. Create a virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

To run the Streamlit application:
```bash
streamlit run app.py
```

## Project Structure

- `app.py` - Main Streamlit application
- `src/`
  - `ranking.py` - Ranking analysis module
  - `trends.py` - Trend analysis module
  - `upload.py` - File upload handling
- `requirements.txt` - Project dependencies

## Dependencies

- streamlit
- pandas
- plotly
- scikit-learn
- openpyxl
- tabulate