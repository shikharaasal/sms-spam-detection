import joblib
import streamlit as st
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "spam_model.pkl"

st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="🛡️",
    layout="centered",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 850px; padding-top: 2.5rem; padding-bottom: 3rem;}
      .hero {padding: 1.5rem 1.6rem; border-radius: 18px;
             background: linear-gradient(135deg, #10233f, #245b83); color: white;
             margin-bottom: 1.3rem;}
      .hero h1 {margin: 0 0 .4rem 0; font-size: 2rem;}
      .hero p {margin: 0; opacity: .88;}
      .note {font-size: .88rem; opacity: .75;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>SMS Spam Detector</h1>
      <p>Machine-learning classification using TF-IDF and Naive Bayes / Logistic Regression.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error("Model file not found: spam_model.pkl")
    st.markdown(
        """
        Add the trained model file to this project folder. In your Colab notebook,
        run the export snippet in `colab_export_snippet.py`, download the file,
        and place it beside `app.py`.
        """
    )
    st.stop()

model = load_model()

st.subheader("Check a message")
message = st.text_area(
    "Paste the SMS message below",
    placeholder="Example: Congratulations! You have won a free prize. Call now...",
    height=160,
    max_chars=10000,
)

if st.button("Detect spam", type="primary", use_container_width=True):
    cleaned_message = message.strip()

    if not cleaned_message:
        st.warning("Enter a message before running the detector.")
    else:
        prediction = int(model.predict([cleaned_message])[0])
        probabilities = model.predict_proba([cleaned_message])[0]
        classes = list(model.classes_)
        spam_index = classes.index(1) if 1 in classes else None

        if spam_index is None:
            st.error("The loaded model does not contain the expected spam class (label 1).")
        else:
            spam_probability = float(probabilities[spam_index])

            st.divider()
            st.subheader("Prediction")

            if prediction == 1:
                st.error("🚨 Likely SPAM")
            else:
                st.success("✅ Likely legitimate (HAM)")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Estimated spam probability", f"{spam_probability:.1%}")
            with col2:
                st.metric("Estimated legitimate probability", f"{1 - spam_probability:.1%}")

            st.progress(spam_probability)
            st.caption(
                "Probabilities are estimates from the trained model, not a guarantee. "
                "The model was trained on SMS messages, so performance may differ on other email types."
            )

with st.expander("About this project"):
    st.write(
        "The notebook trains TF-IDF text-classification pipelines using "
        "Multinomial Naive Bayes and Logistic Regression, evaluates them on a held-out "
        "test set, and selects the model with higher test accuracy."
    )
    st.write(
        "Dataset: SMS Spam Collection. Labels: ham (legitimate) = 0, spam = 1."
    )

st.markdown(
    '<p class="note">Academic demonstration only. Do not rely on this classifier as the sole basis for security decisions.</p>',
    unsafe_allow_html=True,
)
