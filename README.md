# Electricity Bill Prediction (Data Science Project)
Predicts monthly electricity units & bill from household details and appliance usage.

## Pipeline
1. `generate_data.py` - synthetic dataset (5000 households)
2. `train_model.py`   - Linear Regression / Random Forest / Gradient Boosting, saves best model
3. `bill_calculator.py` - appliance wattage x hours -> units -> slab tariff bill
4. `app.py`           - Streamlit web app
5. `predict_cli.py`   - terminal demo

## Run
pip install -r requirements.txt
python generate_data.py
python train_model.py
python predict_cli.py
streamlit run app.py

Tariff slabs are illustrative; edit them in `bill_calculator.py`.
