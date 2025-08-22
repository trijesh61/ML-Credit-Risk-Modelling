
# 💳 Credit Risk Modeling App

A machine learning-based web application to assess the creditworthiness of loan applicants. Built using Python and deployed via Streamlit, the app predicts the probability of default and classifies users into credit risk categories like **Poor**, **Average**, **Good**, and **Excellent**.

[![Streamlit App](https://img.shields.io/badge/Live%20App-Click%20Here-brightgreen?style=for-the-badge&logo=streamlit)](https://ml-credit-risk-modelling-ml.streamlit.app/)


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
- **Scikit-learn**, **XGBoost**
- **Streamlit** for UI
- **Joblib** for model serialization

---

## 📁 Project Structure

```
📦 credit-risk-modelling/
├── app/                  # Streamlit app logic
├── data/                 # Processed datasets (if any)
├── models/               # Saved machine learning models
├── notebooks/            # EDA and model development notebooks
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── SOW_Credit_Risk_Model.pdf  # Statement of Work (optional)
```

---

## 🚀 Getting Started

To run the app locally:

```bash
# 1. Clone the repository
git clone https://github.com/your-username/credit-risk-modelling.git
cd credit-risk-modelling

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app/app.py
```

---

## 📊 Input Parameters

The app expects the following inputs:
- Age, Income, Employment Status
- Credit Utilization, Number of Open Accounts
- Loan Amount, Loan Purpose, and more

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
