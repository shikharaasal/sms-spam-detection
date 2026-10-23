# SMS Spam Detection Using NLP

A student project that classifies SMS messages as **spam** or **legitimate (ham)** using natural language processing and machine learning. The project includes a Google Colab notebook for model development and a Streamlit web app for trying message predictions.

## Live application

**Streamlit app:** Add the deployed app URL here after deployment.

## Project overview

## Open the notebook in Google Colab

[Open the notebook](https://colab.research.google.com/github/shikharaasal/sms-spam-detection/blob/main/Project.ipynb)

The classifier learns patterns in labeled SMS messages and predicts whether a new message is spam or legitimate. Message text is converted into numerical TF-IDF features, then classified using two models:

- Multinomial Naive Bayes
- Logistic Regression

The notebook compares both models using a stratified 80/20 train/test split. In the recorded run, Naive Bayes achieved **96.86% accuracy** and Logistic Regression achieved **96.77% accuracy** on the held-out test set. These results describe this dataset split; performance may differ on newer messages.

## Dataset

The project uses the Kaggle copy of the SMS Spam Collection dataset, containing 5,572 labeled messages:

| Label | Messages |
|---|---:|
| Legitimate (ham) | 4,825 |
| Spam | 747 |

The dataset file in this repository is `SMS Spam Detection - Kaggle Data.csv`.

## Repository contents

| File | Description |
|---|---|
| `Project.ipynb` | Google Colab/Jupyter notebook for loading the data, training and evaluating the models, and trying example predictions |
| `app.py` | Streamlit interface for entering an SMS and viewing its prediction |
| `spam_model.pkl` | Trained model loaded by the Streamlit app |
| `requirements.txt` | Python dependencies for the app |
| `SMS Spam Detection - Kaggle Data.csv` | SMS message dataset used in the project |
| `SMS Spam Detection - Project Report.docx` | Written project report |
| `SMS Spam Detection - Presentation.pptx` | Class presentation |

## Run the Streamlit app locally

1. Install Python 3.10 or 3.11.
2. Download or clone this repository and open a terminal in the project folder.
3. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   On Windows Command Prompt:

   ```bat
   .venv\Scripts\activate.bat
   ```

   On macOS or Linux:

   ```bash
   source .venv/bin/activate
   ```

4. Install the dependencies and start the app:

   ```bash
   python -m pip install -r requirements.txt
   python -m streamlit run app.py
   ```

The app will display a local address in the terminal, usually `http://localhost:8501`.

## Deploy on Streamlit Community Cloud

Deploy this GitHub repository through [Streamlit Community Cloud](https://share.streamlit.io/), selecting `app.py` as the app entry point. The repository must include `requirements.txt` and `spam_model.pkl`. After deployment, replace the Live application placeholder above with the app URL.

## Limitations

- The dataset is historical and may not represent current spam messages or language patterns.
- The dataset has more legitimate messages than spam messages, so accuracy alone does not describe every type of error. The notebook also reports class-level metrics and confusion matrices.
- The Streamlit app is a classroom demonstration and is not connected to a phone or SMS service.
- The model file was serialized with joblib. Only load model files from a source you trust.

## Author

**Shikhar Aasal**  
B.E. (Hons.) Computer Science in Data Science  
AI Applications — BDS-111 (A)  
Chandigarh University
