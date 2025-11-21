# Credit Card Fraud Detection - Streamlit App

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python train_model.py
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

## Features

- **Single Prediction**: Input transaction details and get instant fraud prediction
- **Batch Prediction**: Upload CSV file for bulk fraud detection
- **Interactive Visualizations**: Fraud probability gauges and distributions
- **Model Performance**: High accuracy with SMOTE-balanced Random Forest

## Dataset

Place `fraudTrain.csv` in the project directory before training.
