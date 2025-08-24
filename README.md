
# AI-Powered Surveillance System

## Overview
This repository implements a real-time AI surveillance system that processes video feeds to detect anomalies such as loitering, unusual crowd density, and object abandonment. Utilizing **YOLOv8** for object detection and lightweight temporal logic for anomaly detection, the system provides real-time alerts and visualizations through a **Streamlit** dashboard.

---

## Features
- **Real-time object and person detection** using YOLOv8.
- **Anomaly detection** for:
  - Loitering
  - Object abandonment
  - Unusual crowd density
- **Streamlit dashboard** with:
  - Live video feed and bounding boxes
  - Alert panel with timestamps and event types
  - Downloadable logs in Excel format
- **Configurable thresholds** for adaptable detection parameters

---

## Dataset
- **UCSD Anomaly Detection Dataset**
- **CUHK Avenue Dataset**
- Optional synthetic data using GANs to simulate rare events

---

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/anaraksh/AI_Surveillance.git
   cd AI_Surveillance

2.Create and activate a virtual environment:
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

3.Install the required dependencies:
pip install -r requirements.txt

4.Run the Streamlit app:
streamlit run app.py

