import html
import re

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

MODEL_PATH = Path(__file__).parent / "spam_model.pkl"
DATA_PATH = Path(__file__).parent / "Kaggle Dataset.csv"

# Colors shared by every chart and highlight: red = spam, blue = legitimate (ham)
SPAM_COLOR = "#e34948"
HAM_COLOR = "#2a78d6"
SPAM_RGB = "227, 73, 72"
HAM_RGB = "42, 120, 214"

EXAMPLES = {
    "Obvious spam": (
        "URGENT! Your mobile number has won a £2000 cash award. "
        "Call 09066362231 to claim before it expires."
    ),
    "Everyday message": "Hey, are we still on for dinner tonight? I'll be there around 7.",
    "Legit but spammy": (
        "I won the office raffle! Free lunch on me tomorrow, "
        "call me when you're out of the meeting."
    ),
    "Modern scam": "Your parcel could not be delivered. Reschedule at the link below.",
}

st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon=":material/shield:",
    layout="centered",
)

# Colors and fonts come from .streamlit/config.toml; this covers what the theme can't reach
st.markdown(
    """
    <style>
      .block-container {max-width: 820px; padding-top: 4.5rem; padding-bottom: 4rem;}
      h1, h2, h3 {letter-spacing: -0.01em;}
      h3 {font-size: 1.35rem !important; padding-top: 1.6rem !important;}

      .page-title {font-size: 2.6rem; font-weight: 700; line-height: 1.2; margin: 0 0 .35rem 0;}
      .page-description {color: #787774; font-size: 1.05rem; margin-bottom: 1.4rem;}
      .properties {display: grid; grid-template-columns: 9.5rem 1fr; row-gap: .15rem;
                   font-size: .9rem; margin-bottom: 1.6rem;}
      .properties .key {color: #787774; padding: .3rem 0;}
      .properties .value {padding: .3rem .5rem; border-radius: 4px;}
      .properties .value:hover {background: rgba(55, 53, 47, .06);}

      .callout {display: flex; flex-direction: column; gap: .2rem; padding: 1rem 1.1rem;
                border-radius: 6px; margin: .4rem 0 1rem 0;}
      .callout .title {font-weight: 600; font-size: 1.05rem;}
      .callout.red {background: #fdebec; color: #9f2f2d;}
      .callout.green {background: #edf3ec; color: #2f6a4a;}
      .callout .detail {color: #37352f; opacity: .8; font-size: .92rem;}

      .outcomes {display: grid; grid-template-columns: 7.5rem 1fr 1fr; gap: .5rem; margin: .6rem 0 1rem 0;}
      .outcomes .axis {font-size: .82rem; color: #787774; font-weight: 500; padding-bottom: .1rem;}
      .outcomes .axis.row {align-self: center; padding: 0;}
      .outcome {border-radius: 6px; padding: .9rem 1rem; border: 1px solid transparent;}
      .outcome .count {font-size: 1.8rem; font-weight: 600; line-height: 1.2;}
      .outcome .title {font-weight: 600; font-size: .95rem; margin-top: .15rem;}
      .outcome .detail {font-size: .82rem; color: #787774; margin-top: .1rem;}
      .outcome.good {background: #edf3ec;}
      .outcome.good .count, .outcome.good .title {color: #2f6a4a;}
      .outcome.bad {background: #fdebec;}
      .outcome.bad .count, .outcome.bad .title {color: #9f2f2d;}
      .outcome.neutral {background: #f7f7f5; border-color: #e9e9e7;}
      @media (max-width: 640px) {
        .outcomes {grid-template-columns: 5rem 1fr 1fr;}
        .outcome {padding: .7rem;}
        .outcome .count {font-size: 1.4rem;}
      }

      .highlighted {padding: .9rem 1rem; border-radius: 6px; line-height: 2;
                    background: #f7f7f5; border: 1px solid #e9e9e7;}
      .highlighted span {padding: .1rem .2rem; border-radius: 4px;}
      .legend {font-size: .85rem; color: #787774; margin-bottom: .4rem;}
      .legend span {padding: .1rem .4rem; border-radius: 4px; margin-right: .5rem; color: #37352f;}

      /* Tabs: plain text with a dark underline, like Notion's view switcher */
      .stTabs [data-baseweb="tab-list"] {gap: 1.2rem; border-bottom: 1px solid #e9e9e7;}
      .stTabs [data-baseweb="tab"] {padding: .4rem 0; color: #787774;}
      .stTabs [data-baseweb="tab"] p {font-size: .95rem; font-weight: 500;}
      .stTabs [aria-selected="true"] {color: #37352f;}
      .stTabs [data-baseweb="tab-highlight"],
      .stTabs .react-aria-SelectionIndicator {background-color: #37352f !important; height: 2px;}
      .stTabs [data-baseweb="tab-border"] {display: none;}

      /* Secondary buttons: quiet, gray on hover */
      .stButton button[kind="secondary"] {background: #ffffff; color: #37352f; font-weight: 500;}
      .stButton button[kind="secondary"]:hover {background: rgba(55, 53, 47, .06);
                                                 border-color: #e9e9e7; color: #37352f;}

      /* Metrics: small gray label, medium-weight value */
      [data-testid="stMetricLabel"] p {color: #787774; font-size: .85rem;}
      [data-testid="stMetricValue"] {font-size: 1.9rem; font-weight: 600;}

      /* Expanders look like Notion toggles */
      [data-testid="stExpander"] details {border: none; border-radius: 6px;}
      [data-testid="stExpander"] summary {padding-left: .25rem;}
      [data-testid="stExpander"] summary:hover {background: rgba(55, 53, 47, .06);}

      .sidebar-label {font-size: .78rem; font-weight: 600; color: #91918e; margin: .5rem 0 .6rem 0;}
      .footer {margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #e9e9e7;
               font-size: .85rem; color: #9b9a97;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="page-title">SMS Spam Detector</div>
    <div class="page-description">
      Paste a text message to see whether a machine-learning model thinks it is spam, and why.
    </div>
    """,
    unsafe_allow_html=True,
)


def callout(title, detail, tone):
    st.markdown(
        f'<div class="callout {tone}"><div class="title">{title}</div>'
        f'<div class="detail">{detail}</div></div>',
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    # Same loading and label encoding as the notebook. The repo copy is UTF-8;
    # the original Kaggle download is latin-1.
    try:
        df = pd.read_csv(DATA_PATH, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(DATA_PATH, encoding="latin-1")
    if {"v1", "v2"}.issubset(df.columns):
        df = df[["v1", "v2"]].copy()
    else:
        df = df.iloc[:, :2].copy()
    df.columns = ["label", "text"]
    df = df.dropna(subset=["label", "text"]).copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower().map({"ham": 0, "spam": 1})
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].astype(str)
    return df


@st.cache_data
def test_set_predictions():
    # Recreates the notebook's 80/20 split, so these are messages the model never saw in training
    df = load_dataset()
    _, X_test, _, y_test = train_test_split(
        df["text"], df["label"], test_size=0.20, random_state=42, stratify=df["label"]
    )
    model = load_model()
    spam_index = list(model.classes_).index(1)
    return pd.DataFrame({
        "text": X_test.values,
        "label": y_test.values,
        "spam_probability": model.predict_proba(X_test)[:, spam_index],
    })


def word_weights(model):
    """How strongly each vocabulary word pushes towards spam (positive) or ham (negative).

    The model's spam log-odds for a message is exactly bias + sum(tfidf_value * weight).
    """
    vectorizer, classifier = model.steps[0][1], model.steps[-1][1]
    classes = list(classifier.classes_)
    if hasattr(classifier, "feature_log_prob_"):  # Multinomial Naive Bayes
        spam, ham = classes.index(1), classes.index(0)
        weights = classifier.feature_log_prob_[spam] - classifier.feature_log_prob_[ham]
        bias = classifier.class_log_prior_[spam] - classifier.class_log_prior_[ham]
    elif hasattr(classifier, "coef_"):  # Logistic Regression
        weights, bias = classifier.coef_[0], classifier.intercept_[0]
    else:
        return None
    return vectorizer, weights, bias


def explain(model, message):
    vectorizer, weights, bias = word_weights(model)
    features = vectorizer.transform([message])
    names = vectorizer.get_feature_names_out()
    contributions = {
        names[i]: float(value * weights[i])
        for i, value in zip(features.indices, features.data)
    }
    tokens = set(vectorizer.build_analyzer()(message))
    known = tokens & set(vectorizer.vocabulary_)
    return contributions, bias, len(known), len(tokens)


def highlight(message, contributions, token_pattern):
    largest = max((abs(c) for c in contributions.values()), default=0) or 1
    parts, last = [], 0
    for match in re.finditer(token_pattern, message):
        contribution = contributions.get(match.group().lower())
        if not contribution:
            continue
        rgb = SPAM_RGB if contribution > 0 else HAM_RGB
        alpha = 0.15 + 0.55 * abs(contribution) / largest
        parts.append(html.escape(message[last:match.start()]))
        parts.append(
            f'<span style="background: rgba({rgb}, {alpha:.2f})" '
            f'title="{contribution:+.2f}">{html.escape(match.group())}</span>'
        )
        last = match.end()
    parts.append(html.escape(message[last:]))
    return "".join(parts).replace("\n", "<br>")


def contribution_chart(contributions, limit=10):
    top = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)[:limit]
    data = pd.DataFrame(top, columns=["word", "contribution"])
    data["direction"] = np.where(data["contribution"] > 0, "Towards spam", "Towards legitimate")
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusEnd=4, size=16)
        .encode(
            x=alt.X("contribution:Q", title="Push on the verdict (log-odds)"),
            y=alt.Y("word:N", sort=alt.EncodingSortField("contribution", op="max", order="descending"), title=None),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(domain=["Towards spam", "Towards legitimate"], range=[SPAM_COLOR, HAM_COLOR]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[alt.Tooltip("word:N"), alt.Tooltip("contribution:Q", format="+.2f"), "direction:N"],
        )
        .properties(height=90 + 30 * len(data))
    )


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
if 1 not in list(model.classes_):
    st.error("The loaded model does not contain the expected spam class (label 1).")
    st.stop()

# Notion-style page properties
classifier_name = {
    "MultinomialNB": "Multinomial Naive Bayes",
    "LogisticRegression": "Logistic Regression",
}.get(type(model.steps[-1][1]).__name__, type(model.steps[-1][1]).__name__)
properties = [
    ("Model", classifier_name),
    ("Text features", "TF-IDF word weights"),
    ("Training data", "SMS Spam Collection"),
]
if DATA_PATH.exists():
    held_out = test_set_predictions()
    held_out_accuracy = ((held_out["spam_probability"] >= 0.5) == (held_out["label"] == 1)).mean()
    properties.append(("Test accuracy", f"{held_out_accuracy:.1%} on {len(held_out):,} unseen messages"))
st.markdown(
    '<div class="properties">'
    + "".join(f'<div class="key">{key}</div><div class="value">{value}</div>' for key, value in properties)
    + "</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="sidebar-label">Settings</div>', unsafe_allow_html=True)
    threshold = st.slider(
        "Spam threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help="A message is flagged as spam when its spam probability reaches this value.",
    )
    st.caption(
        "Lower it to catch more spam at the cost of more false alarms; raise it to only flag "
        "messages the model is very sure about. The **Model performance** tab shows the trade-off."
    )

try_tab, performance_tab, data_tab = st.tabs(["Try it", "Model performance", "About the data"])


def use_example(text):
    st.session_state.message = text
    st.session_state.checked = text


def check_message():
    st.session_state.checked = st.session_state.message.strip()


with try_tab:
    st.subheader("Check a message")
    st.caption("Try an example:")
    example_columns = st.columns(len(EXAMPLES))
    for column, (name, text) in zip(example_columns, EXAMPLES.items()):
        column.button(name, on_click=use_example, args=(text,), width="stretch")

    message = st.text_area(
        "Paste the SMS message below",
        key="message",
        placeholder="Example: Congratulations! You have won a free prize. Call now...",
        height=160,
        max_chars=10000,
    )
    clicked = st.button("Detect spam", type="primary", on_click=check_message)

    checked = st.session_state.get("checked", "")
    if clicked and not checked:
        st.warning("Enter a message before running the detector.")
    elif checked and checked == message.strip():
        spam_index = list(model.classes_).index(1)
        spam_probability = float(model.predict_proba([checked])[0][spam_index])

        st.subheader("Prediction")

        if spam_probability >= threshold:
            callout("Likely spam", f"The model flags messages at {threshold:.0%} spam probability or above.", "red")
        else:
            callout(
                "Likely legitimate (ham)",
                f"The model flags messages at {threshold:.0%} spam probability or above.",
                "green",
            )

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Estimated spam probability", f"{spam_probability:.1%}")
        with col2:
            st.metric("Estimated legitimate probability", f"{1 - spam_probability:.1%}")

        st.progress(spam_probability)
        st.caption(
            "You can change the spam threshold in the sidebar. Probabilities are estimates from the trained model, not a guarantee. "
            "The model was trained on SMS messages, so performance may differ on other message types."
        )

        if word_weights(model) is not None:
            contributions, bias, known, total = explain(model, checked)

            st.subheader("Why the model decided this")
            if total == 0:
                st.info(
                    "This message has no words the model can use (very common words like "
                    "“the” and “is” are ignored), so the verdict reflects only how common spam "
                    "was in the training data."
                )
            elif known == 0:
                st.warning(
                    f"The model recognised none of the {total} meaningful words in this message, "
                    "so the verdict reflects only how common spam was in the training data."
                )
            else:
                if known / total < 0.5:
                    st.warning(
                        f"The model recognised only {known} of {total} meaningful words, "
                        "so this verdict rests on limited evidence."
                    )
                else:
                    st.caption(f"The model recognised {known} of {total} meaningful words in this message.")

                st.markdown(
                    '<div class="legend">'
                    f'<span style="background: rgba({SPAM_RGB}, .45)">pushes towards spam</span>'
                    f'<span style="background: rgba({HAM_RGB}, .45)">pushes towards legitimate</span>'
                    "</div>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div class="highlighted">'
                    f"{highlight(checked, contributions, model.steps[0][1].token_pattern)}</div>",
                    unsafe_allow_html=True,
                )
                st.write("")
                st.altair_chart(contribution_chart(contributions))
                st.caption(
                    "Before reading any words the model already leans legitimate, because most "
                    f"training messages were (starting push {bias:+.2f}). Each recognised word then "
                    "adds its push; the total decides the probability. Unhighlighted words were "
                    "either ignored as too common or never seen during training."
                )

with performance_tab:
    if not DATA_PATH.exists():
        st.info(f"Add `{DATA_PATH.name}` beside `app.py` to see how the model performs on held-out messages.")
    else:
        results = test_set_predictions()
        is_spam = results["label"] == 1
        flagged = results["spam_probability"] >= threshold
        true_positive = int((is_spam & flagged).sum())
        false_positive = int((~is_spam & flagged).sum())
        false_negative = int((is_spam & ~flagged).sum())
        true_negative = int((~is_spam & ~flagged).sum())
        accuracy = (true_positive + true_negative) / len(results)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / is_spam.sum()

        st.subheader("How well does it work?")
        st.write(
            f"Measured on **{len(results):,} test messages** the model never saw during training "
            "(the same 80/20 split as the notebook), "
            f"using the current spam threshold of **{threshold:.0%}**."
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy:.1%}", help="Share of all messages classified correctly.")
        col2.metric(
            "Spam caught (recall)",
            f"{recall:.1%}",
            help="Of the real spam messages, the share the model flagged.",
        )
        col3.metric(
            "Spam flags correct (precision)",
            f"{precision:.1%}",
            help="Of the messages the model flagged as spam, the share that really were spam.",
        )

        baseline = (~is_spam).mean()
        st.info(
            f"**Why accuracy alone is misleading:** {baseline:.1%} of test messages are legitimate, "
            f"so a 'model' that never flags anything would already score {baseline:.1%} accuracy "
            "while catching zero spam. Recall and precision show what actually matters."
        )

        st.subheader("Where it gets things right and wrong")
        legit_total = true_negative + false_positive

        def outcome_card(count, title, detail, correct):
            tone = "good" if correct else ("bad" if count else "neutral")
            return (
                f'<div class="outcome {tone}"><div class="count">{count:,}</div>'
                f'<div class="title">{title}</div><div class="detail">{detail}</div></div>'
            )

        spam_total = true_positive + false_negative
        st.markdown(
            '<div class="outcomes">'
            '<div></div><div class="axis">Model said spam</div><div class="axis">Model said legitimate</div>'
            '<div class="axis row">Actually spam</div>'
            + outcome_card(true_positive, "Spam caught", f"{true_positive / spam_total:.1%} of real spam", True)
            + outcome_card(false_negative, "Spam missed", f"{false_negative / spam_total:.1%} of real spam slipped through", False)
            + '<div class="axis row">Actually legitimate</div>'
            + outcome_card(false_positive, "False alarms", f"{false_positive / legit_total:.1%} of real messages wrongly flagged", False)
            + outcome_card(true_negative, "Correctly allowed", f"{true_negative / legit_total:.1%} of real messages", True)
            + "</div>",
            unsafe_allow_html=True,
        )

        st.subheader("Choosing the threshold")
        thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
        curve = []
        for value in thresholds:
            hits = results["spam_probability"] >= value
            caught = (is_spam & hits).sum()
            curve.append({"threshold": value, "metric": "Spam caught (recall)", "value": caught / is_spam.sum()})
            curve.append({
                "threshold": value,
                "metric": "Spam flags correct (precision)",
                "value": caught / hits.sum() if hits.sum() else 1.0,
            })
        curve = pd.DataFrame(curve)
        hover = alt.selection_point(fields=["threshold"], nearest=True, on="pointerover", empty=False)
        lines = alt.Chart(curve).mark_line(strokeWidth=2).encode(
            x=alt.X("threshold:Q", title="Spam threshold", axis=alt.Axis(format="%")),
            y=alt.Y("value:Q", title=None, axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color(
                "metric:N",
                scale=alt.Scale(range=["#2a78d6", "#eb6834"]),
                legend=alt.Legend(title=None, orient="top", labelLimit=0),
            ),
        )
        points = lines.mark_point(filled=True, size=70).encode(
            opacity=alt.condition(hover, alt.value(1), alt.value(0)),
            tooltip=[
                alt.Tooltip("threshold:Q", format=".0%"),
                "metric:N",
                alt.Tooltip("value:Q", format=".1%"),
            ],
        ).add_params(hover)
        current = alt.Chart(pd.DataFrame({"threshold": [threshold]})).mark_rule(
            strokeDash=[4, 4], color="gray"
        ).encode(x="threshold:Q")
        st.altair_chart((lines + points + current).properties(height=280))
        st.caption(
            "The dashed line is the current threshold. Moving it left catches more spam but "
            "eventually starts flagging real messages; moving it right does the opposite."
        )

        missed = results[is_spam & ~flagged].sort_values("spam_probability", ascending=False)
        false_alarms = results[~is_spam & flagged].sort_values("spam_probability", ascending=False)
        with st.expander(f"Spam the model missed ({len(missed)})"):
            st.dataframe(
                missed[["text", "spam_probability"]].rename(
                    columns={"text": "Message", "spam_probability": "Spam probability"}
                ),
                hide_index=True,
                width="stretch",
                column_config={"Spam probability": st.column_config.NumberColumn(format="%.3f")},
            )
        with st.expander(f"Legitimate messages flagged as spam ({len(false_alarms)})"):
            if false_alarms.empty:
                st.write("None at this threshold.")
            else:
                st.dataframe(
                    false_alarms[["text", "spam_probability"]].rename(
                        columns={"text": "Message", "spam_probability": "Spam probability"}
                    ),
                    hide_index=True,
                    width="stretch",
                    column_config={"Spam probability": st.column_config.NumberColumn(format="%.3f")},
                )

with data_tab:
    st.subheader("About this project")
    st.write(
        "The notebook trains TF-IDF text-classification pipelines using "
        "Multinomial Naive Bayes and Logistic Regression, evaluates them on a held-out "
        "test set, and selects the model with higher test accuracy."
    )

    if not DATA_PATH.exists():
        st.info(f"Add `{DATA_PATH.name}` beside `app.py` to explore the training data.")
    else:
        df = load_dataset()
        df["length"] = df["text"].str.len()
        df["class"] = df["label"].map({0: "Legitimate", 1: "Spam"})

        st.subheader("The dataset")
        st.write(f"The SMS Spam Collection: **{len(df):,} real text messages**, each labelled by hand.")
        col1, col2 = st.columns(2)
        counts = df["class"].value_counts()
        for column, name, label in [(col1, "Legitimate", "Legitimate (ham)"), (col2, "Spam", "Spam")]:
            column.metric(label, f"{counts[name]:,}")
            column.caption(f"{counts[name] / len(df):.1%} of messages")

        st.subheader("Spam messages tend to be longer")
        medians = df.groupby("class")["length"].median()
        st.write(
            f"The typical spam message is **{medians['Spam']:.0f} characters**, "
            f"versus **{medians['Legitimate']:.0f}** for a legitimate one."
        )
        trimmed = df[df["length"] <= 300]
        for name, color in [("Legitimate", HAM_COLOR), ("Spam", SPAM_COLOR)]:
            histogram = alt.Chart(trimmed[trimmed["class"] == name]).mark_bar(
                cornerRadiusEnd=2, binSpacing=2, color=color
            ).encode(
                x=alt.X(
                    "length:Q",
                    bin=alt.Bin(step=10, extent=[0, 300]),
                    scale=alt.Scale(domain=[0, 300]),
                    title="Message length (characters)",
                ),
                y=alt.Y("count():Q", title="Messages"),
                tooltip=[
                    alt.Tooltip("length:Q", bin=alt.Bin(step=10, extent=[0, 300]), title="Length"),
                    alt.Tooltip("count():Q", title="Messages"),
                ],
            ).properties(title=name, height=240)
            st.altair_chart(histogram)
        st.caption(f"Messages over 300 characters ({(df['length'] > 300).sum()}) are left out to keep the chart readable.")

        st.subheader("Most common words")
        st.write("Share of messages in each class that contain the word (very common words like “the” are excluded).")
        counter = CountVectorizer(stop_words="english", binary=True)
        word_columns = st.columns(2)
        for column, (name, color) in zip(word_columns, [("Spam", SPAM_COLOR), ("Legitimate", HAM_COLOR)]):
            texts = df.loc[df["class"] == name, "text"]
            matrix_counts = counter.fit_transform(texts)
            shares = np.asarray(matrix_counts.sum(axis=0)).ravel() / len(texts)
            top = np.argsort(shares)[::-1][:12]
            words = pd.DataFrame({"word": counter.get_feature_names_out()[top], "share": shares[top]})
            chart = alt.Chart(words).mark_bar(cornerRadiusEnd=4, size=14, color=color).encode(
                x=alt.X("share:Q", title="Share of messages", axis=alt.Axis(format="%")),
                y=alt.Y("word:N", sort="-x", title=None),
                tooltip=[alt.Tooltip("word:N"), alt.Tooltip("share:Q", format=".1%", title="Share of messages")],
            ).properties(title=name, height=400)
            column.altair_chart(chart)
        st.caption(
            "Note the different scales: spam reuses the same words far more often. "
            "“gt” and “lt” come from “&lt;#&gt;” placeholders that replaced numbers in the original dataset."
        )

        st.subheader("Browse messages")
        browse_class = st.radio("Show", ["Spam", "Legitimate", "Both"], horizontal=True)
        if st.button("Shuffle"):
            st.session_state.sample_seed = int(np.random.randint(0, 1_000_000))
        pool = df if browse_class == "Both" else df[df["class"] == browse_class]
        sample = pool.sample(8, random_state=st.session_state.get("sample_seed", 0))
        st.dataframe(
            sample[["class", "text"]].rename(columns={"class": "Label", "text": "Message"}),
            hide_index=True,
            width="stretch",
        )

st.markdown(
    '<div class="footer">Academic demonstration only. Do not rely on this classifier as the sole basis for security decisions.</div>',
    unsafe_allow_html=True,
)
