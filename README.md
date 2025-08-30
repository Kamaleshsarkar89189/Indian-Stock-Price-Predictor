# 📈 Indian Stock Price Prediction

A **Streamlit-powered machine learning project** for predicting Indian stock prices.
The system includes modular components for **data ingestion, transformation, training, and prediction**, making it scalable and production-ready.

---

## 🚀 Features

* 📊 **Fetch & preprocess stock data** (Yahoo Finance API)
* 🔄 **Data transformation pipeline** with cleaning, scaling, and feature engineering
* 🤖 **Model training pipeline** with multiple ML models
* 🎯 **Prediction pipeline** for forecasting stock prices
* 📉 Interactive **Streamlit dashboard** for visualization and predictions
* 📝 **Logging & exception handling** for better debugging

---

## 🗂 Project Structure

```
src/
│── Indian_Stock_Price_Prediction/
│   ├── components/
│   │   ├── data_ingestion.py       # Fetches stock data
│   │   ├── data_transformation.py  # Cleans & transforms data
│   │   ├── model_trainer.py        # Trains ML models
│   │   ├── __init__.py
│   │
│   ├── pipelines/
│   │   ├── training_pipeline.py    # End-to-end training flow
│   │   ├── prediction_pipeline.py  # End-to-end prediction flow
│   │   ├── __init__.py
│   │
│   ├── utils.py                    # Helper utilities
│   ├── plots.py                    # Visualization functions
│   ├── logger.py                   # Logging setup
│   ├── exception.py                # Custom exception handling
│   ├── __init__.py
│
│── app.py                          # Streamlit app entry point
│── main.py                         # Orchestrates pipelines
│── requirements.txt                # Dependencies
│── README.md                       # Documentation
```




## 📊 Example Outputs

* Historical stock price trends
* Train/test evaluation metrics (MSE, RMSE, R²)
* Predicted price charts
* Comparison of **actual vs predicted** prices

---

## 🛠 Tech Stack

* **Python 3.9+**
* **Streamlit** – UI framework
* **scikit-learn** – ML models
* **yfinance** – Stock market data
* **Plotly** – Visualization

---

## 🔮 Future Improvements

* [ ] Add **Dockerfile** for deployment
* [ ] Integrate deep learning (LSTM/Transformers)
* [ ] Cloud deployment (AWS/GCP/Streamlit Cloud)
* [ ] Multiple stock comparison module

---

## 👤 Author

**Piyush Agarwal**  
📌 *Learning • Building • Growing*

🔗 [Streamlit App](https://indianstockpriceprediction.streamlit.app/) | [LinkedIn](https://linkedin.com/) | [GitHub](https://github.com/)

---



