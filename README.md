
# 💳 Credit Risk Modeling App

A machine learning-based web application to assess the creditworthiness of loan applicants. Built using Python and deployed via Streamlit, the app predicts the probability of default and classifies users into credit risk categories like **Poor**, **Average**, **Good**, and **Excellent**.

[![Streamlit App](https://img.shields.io/badge/Live%20App-Click%20Here-brightgreen?style=for-the-badge&logo=streamlit)](https://ml-credit-risk-modelling.streamlit.app/)


---

## 📌 Key Features

- Predicts loan default probability based on user input
- Classifies applicants into credit risk categories
- User-friendly Streamlit interface
- Real-time inference using a trained ML model
- Easy to use for loan officers or credit analysts

---

## 🛠️ Technologies Used

- **Python**
- **Pandas**, **NumPy**
- **Scikit-learn**
- **Streamlit** for UI
- **Joblib** for model serialization

---

## 📁 Project Structure

```
📦 ML-Credit-Risk-Modelling/
├── app/                   # Streamlit app (entry: app/main.py)
│   ├── __init__.py
│   ├── main.py
│   └── prediction_helper.py
├── artifacts/             # Saved model artifact(s)
│   └── model_data.joblib
├── dataset/               # Source CSVs (optional for reference)
│   ├── bureau_data.csv
│   ├── customers.csv
│   └── loans.csv
├── credit_risk_model.ipynb
├── credit_risk_model_python_code.ipynb
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 🚀 Getting Started (Local)

To run the app locally:

```bash
# 1) Clone the repository
git clone https://github.com/<your-username>/ML-Credit-Risk-Modelling.git
cd ML-Credit-Risk-Modelling

# 2) Create/activate a virtual env (recommended)
# python -m venv .venv && source .venv/bin/activate   # macOS/Linux
# python -m venv .venv && .venv\Scripts\activate      # Windows

# 3) Install dependencies
pip install -r requirements.txt

# 4) Run the Streamlit app
streamlit run app/main.py
```

---

## ☁️ Deploy on Streamlit Community Cloud

1. Push your repo to GitHub.
2. Go to share.streamlit.io, select the repo/branch.
3. Set Main file path to `app/main.py`.
4. Ensure `requirements.txt` exists at repo root. Deploy.

---

## 📊 Input Parameters

The app expects the following inputs:
- Age, Income, Loan Amount, Loan Tenure
- Credit Utilization, Delinquency Ratio, Avg DPD, Open Accounts
- Residence Type, Loan Purpose, Loan Type

---

## 🧠 Model

The backend model uses historical loan data to learn patterns associated with defaults. The model outputs:
- A probability of default
- A credit score classification

---

## 📌 License

This project is for educational/demo purposes. Feel free to fork and modify it for your own use.

---

## 🙋‍♂️ Contact

For any questions or suggestions, feel free to reach out via [LinkedIn](https://linkedin.com/in/trijesh-kondapuram) or raise an issue in the repo.
