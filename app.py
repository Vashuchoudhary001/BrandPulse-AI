
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

#page
st.set_page_config(
    page_title="BrandPulse AI",
    page_icon="📈",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# Existing Classical ML artifacts
CLASSICAL_PATH = MODEL_DIR / "logistic_regression.pkl"
TFIDF_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"

# Expected LSTM artifacts 
LSTM_PATH = MODEL_DIR / "lstm_sentiment_model.keras"
TOKENIZER_PATH = MODEL_DIR / "lstm_tokenizer.pkl"


# designnn
st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(ellipse at 50% 0%,
        #001b57 0%, #020716 45%, #000000 100%);
    color: #eef6ff;
}

.block-container {
    max-width: 1250px;
    padding-top: 0 !important;
    margin-top: 0 !important;
}

.brand-header {
    text-align: center;
    padding: 30px 15px;
    margin-top: 60px;
    margin-bottom: 28px;
    background: linear-gradient(135deg, #030d24, #071c43);
    border: 1px solid #087dff;
    border-radius: 20px;
    box-shadow: 0 0 24px #005dff66, inset 0 0 20px #004cff22;
}

.brand-header h1 {
    color: #ffffff;
    font-size: 42px;
    margin: 0;
    text-shadow: 0 0 18px #168aff;
}

.brand-header p {
    color: #7fc9ff;
    margin-bottom: 0;
}

.model-card {
    background: linear-gradient(145deg, #06142d, #020817);
    border: 1px solid #087dff;
    border-radius: 17px;
    padding: 20px;
    box-shadow: 0 0 15px #005dff44;
    min-height: 125px;
    margin-bottom: 12px;
}

.model-card h3 {
    color: #70c5ff;
    margin-top: 0;
}

.model-card p {
    color: #d5e7ff;
}

.stTextArea textarea {
    background: #030d20 !important;
    color: white !important;
    border: 1px solid #087dff !important;
    border-radius: 10px !important;
}

.stButton > button {
    width: 100%;
    min-height: 48px;
    color: white;
    font-weight: 700;
    background: linear-gradient(90deg, #075bff, #00aaff);
    border: 1px solid #40cfff;
    border-radius: 10px;
    box-shadow: 0 0 12px #006dff55;
}

.stButton > button:hover {
    color: white;
    border: 1px solid #ffffff;
    box-shadow: 0 0 22px #008cffaa;
}

div[data-testid="stMetric"] {
    background: #06152d;
    border: 1px solid #087dff;
    border-radius: 12px;
    padding: 15px;
}

h1, h2, h3, p, label {
    color: #eef6ff;
}

hr {
    border-color: #143b72;
}
</style>
""", unsafe_allow_html=True)


# classic model reloading
@st.cache_resource
def load_classical():
    model = joblib.load(CLASSICAL_PATH)
    vectorizer = joblib.load(TFIDF_PATH)
    return model, vectorizer


try:
    classical_model, tfidf = load_classical()
except Exception as error:
    st.error("Classical ML model could not be loaded.")
    st.code(str(error))
    st.stop()


#lstm model
@st.cache_resource
def load_lstm():
    from tensorflow.keras.models import load_model

    lstm_model = load_model(LSTM_PATH)
    tokenizer = joblib.load(TOKENIZER_PATH)

    return lstm_model, tokenizer


lstm_available = (
    LSTM_PATH.exists() and TOKENIZER_PATH.exists()
)

lstm_model = None
tokenizer = None

if lstm_available:
    try:
        lstm_model, tokenizer = load_lstm()
    except Exception:
        lstm_available = False


#predic fun
def predict_classical(text):
    vector = tfidf.transform([text])
    prediction = classical_model.predict(vector)[0]

    confidence = None
    if hasattr(classical_model, "predict_proba"):
        confidence = float(
            classical_model.predict_proba(vector).max()
        ) * 100

    return str(prediction).lower(), confidence


def predict_lstm(text):
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    # IMPORTANT: maxlen must match your LSTM training code.
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=50, padding="post",
                           truncating="post")

    output = np.asarray(lstm_model.predict(padded, verbose=0))
    scores = output.reshape(-1)

    
    # Class order must match the LSTM training label encoder.
    if len(scores) != 3:
        raise ValueError(
            "Expected 3 output scores. Check your LSTM output layer."
        )

    labels = ["negative", "neutral", "positive"]
    index = int(np.argmax(scores))

    return labels[index], float(scores[index]) * 100


def show_prediction(label, confidence):
    if label == "positive":
        st.success("😊 Positive sentiment")
    elif label == "negative":
        st.error("☹️ Negative sentiment")
    else:
        st.info("😐 Neutral sentiment")

    if confidence is not None:
        st.metric("Model confidence estimate", f"{confidence:.1f}%")


#header
st.markdown("""
<div class="brand-header">
    <h1>📊 BrandPulse AI</h1>
    <p>Twitter Sentiment Intelligence</p>
    <p>Classical Machine Learning · Deep Learning</p>
</div>
""", unsafe_allow_html=True)

# for  simulated live tweet stream

st.markdown("---")
st.markdown("## 📡 Live Tweet Stream")
st.caption(
    "Demo mode: Generate sample tweets and analyze them "
    "with the Classical ML model."
)

# Sample tweets simulation  sss
sample_tweets = [
    "I love the excellent service and friendly staff!",
    "My flight was delayed again. Very disappointing.",
    "The flight was okay, nothing special.",
    "Amazing experience! Everything was perfect.",
    "The customer support was terrible.",
    "My journey was average.",
    "Very happy with the quick service.",
    "The seats were uncomfortable and the food was bad.",
    "Everything was fine during my trip.",
    "Thank you for making my journey pleasant!"
]

# Save  the feed data between Streamlit reruns
if "live_tweets" not in st.session_state:
    st.session_state.live_tweets = []

if "live_index" not in st.session_state:
    st.session_state.live_index = 0

# Feed controls
start_col, clear_col = st.columns(2)

with start_col:
    generate_clicked = st.button(
        "📥 Simulate Incoming Tweet",
        key="generate_live_tweet",
        use_container_width=True
    )

with clear_col:
    clear_clicked = st.button(
        "🗑️ Clear Feed",
        key="clear_live_feed",
        use_container_width=True
    )

# Adding  one new simulated tweet 
if generate_clicked:
    tweet_text = sample_tweets[
        st.session_state.live_index % len(sample_tweets)
    ]

    sentiment, confidence = predict_classical(tweet_text)

    now = datetime.now()

    tweet_record = {
    "text": tweet_text,
    "sentiment": sentiment,
    "confidence": confidence,
    "time": now.strftime("%I:%M:%S %p"),
    "timestamp": now
 }

    st.session_state.live_tweets.append(tweet_record)
    st.session_state.live_index += 1

    # only  20 tweets rakhenge
    st.session_state.live_tweets = (
        st.session_state.live_tweets[-20:]
    )

if clear_clicked:
    st.session_state.live_tweets = []
    st.session_state.live_index = 0

# Display newest tweets first
if st.session_state.live_tweets:
    for tweet in reversed(st.session_state.live_tweets):
        with st.container(border=True):
            st.caption("🕒 " + tweet["time"])
            st.write(tweet["text"])

            if tweet["sentiment"] == "positive":
                st.success("😊 POSITIVE ")
            elif tweet["sentiment"] == "negative":
                st.error("☹️ NEGATIVE")
            else:
                st.info("😐 NEUTRAL")

            if tweet["confidence"] is not None:
                st.caption(
                    "Model confidence estimate: "
                    f'{tweet["confidence"]:.1f}%'
                )
else:
    st.info(
        "No incoming tweets yet. Click the button to start the demo."
    )
#sentimetn distribution

st.markdown("---")
st.markdown("## 📊 Live Sentiment Distribution")

if st.session_state.live_tweets:

    # to Get the sentiments from all tweets currently in the feed
    sentiment_counts = pd.Series(
        [tweet["sentiment"] for tweet in st.session_state.live_tweets]
    ).value_counts()

    # Keep a consistent order, even if a sentiment has zero tweets
    sentiment_counts = sentiment_counts.reindex(
        ["positive", "neutral", "negative"],
        fill_value=0
    )

    total_tweets = int(sentiment_counts.sum())

    # for percentages
    sentiment_percentages = (
        sentiment_counts / total_tweets * 100
    )

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        st.metric("Tweets in feed", total_tweets)
        st.metric(
            "😊 Positive",
            f'{sentiment_percentages["positive"]:.1f}%'
        )
        st.metric(
            "😐 Neutral",
            f'{sentiment_percentages["neutral"]:.1f}%'
        )
        st.metric(
            "☹️ Negative",
            f'{sentiment_percentages["negative"]:.1f}%'
        )

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.pie(
            sentiment_counts.values,
            labels=["Positive", "Neutral", "Negative"],
            autopct="%1.1f%%",
            startangle=90
        )

        ax.set_title("Sentiment Breakdown")
        ax.axis("equal")

        st.pyplot(fig)
        plt.close(fig)

else:
    st.info("Generate some tweets to see the sentiment chart.")

  #sentiment over time

st.markdown("---")
st.markdown("## 📈 Sentiment Trend Over Time")

if st.session_state.live_tweets:

    # Convert live tweets into a DataFrame
    trend_df = pd.DataFrame(st.session_state.live_tweets)

    # Count tweets in the order they arrived
    trend_df["tweet_number"] = range(1, len(trend_df) + 1)

    # Separate each sentiment
    for sentiment in ["positive", "neutral", "negative"]:

        sentiment_values = (
            trend_df["sentiment"] == sentiment
        ).astype(int).cumsum()

        plt.plot(
            trend_df["tweet_number"],
            sentiment_values,
            marker="o",
            label=sentiment.capitalize()
        )

    plt.xlabel("Incoming Tweet Number")
    plt.ylabel("Cumulative Tweet Count")
    plt.title("How Sentiments Change as Tweets Arrive")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    st.pyplot(plt.gcf())
    plt.close()

else:
    st.info("Generate some tweets to see the sentiment trend.")




#model panel separate
left, right = st.columns(2, gap="large")

with left:
    st.markdown("""
    <div class="model-card">
        <h3>📊 CLASSICAL ML</h3>
        <p><b>TF-IDF + Logistic Regression</b></p>
        <p>Traditional machine learning for sentiment classification.</p>
    </div>
    """, unsafe_allow_html=True)

    classical_text = st.text_area(
        "Enter a tweet for Classical ML",
        placeholder="Example: I love this service!",
        height=120,
        key="classical_text"
    )

    if st.button("Analyze with Classical ML"):
        if not classical_text.strip():
            st.warning("Please enter a tweet.")
        else:
            label, confidence = predict_classical(classical_text)
            show_prediction(label, confidence)


with right:
    st.markdown("""
    <div class="model-card">
        <h3>🧠 DEEP LEARNING</h3>
        <p><b>LSTM Neural Network</b></p>
        <p>A recurrent neural network for sequence-based text analysis.</p>
    </div>
    """, unsafe_allow_html=True)

    lstm_text = st.text_area(
        "Enter a tweet for LSTM",
        placeholder="Example: The experience was wonderful!",
        height=120,
        key="lstm_text"
    )

    if not lstm_available:
        st.warning(
            "LSTM model is not ready yet. Train and save the LSTM model "
            "and tokenizer before using this prediction button."
        )

    if st.button("Analyze with LSTM", disabled=not lstm_available):
        if not lstm_text.strip():
            st.warning("Please enter a tweet.")
        else:
            try:
                label, confidence = predict_lstm(lstm_text)
                show_prediction(label, confidence)
            except Exception as error:
                st.error("LSTM prediction failed. Check training settings.")
                st.code(str(error))


#for bulk csv
st.markdown("---")
st.markdown("## 📂 Bulk CSV Analysis")
st.write("Upload a CSV containing tweets for batch analysis.")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"],
    key="bulk_csv"
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        if data.empty or len(data.columns) == 0:
            st.warning("The uploaded CSV is empty.")
        else:
            st.dataframe(data.head(5), use_container_width=True)

            columns = list(data.columns)
            preferred = [
                c for c in columns
                if c.lower().strip() in
                ["text", "tweet", "tweet_text", "clean_text"]
            ]

            default_index = (
                columns.index(preferred[0]) if preferred else 0
            )

            text_column = st.selectbox(
                "Select the tweet text column",
                columns,
                index=default_index
            )

            if st.button("Analyze CSV with Classical ML"):
                tweets = data[text_column].fillna("").astype(str)
                mask = tweets.str.strip().ne("")

                if not mask.any():
                    st.warning("No non-empty tweets found.")
                else:
                    vectors = tfidf.transform(tweets[mask])
                    predictions = classical_model.predict(vectors)

                    results = data.copy()
                    results["predicted_sentiment"] = "not analyzed"
                    results.loc[mask, "predicted_sentiment"] = predictions

                    counts = pd.Series(predictions).value_counts()
                    counts = counts.reindex(
                        ["positive", "neutral", "negative"],
                        fill_value=0
                    )

                    st.markdown("### Sentiment Summary")
                    a, b, c = st.columns(3)
                    a.metric("😊 Positive", int(counts["positive"]))
                    b.metric("😐 Neutral", int(counts["neutral"]))
                    c.metric("☹️ Negative", int(counts["negative"]))

                    chart, ax = plt.subplots()
                    counts.plot(kind="bar", ax=ax)
                    ax.set_xlabel("Sentiment")
                    ax.set_ylabel("Number of tweets")
                    ax.set_title("Classical ML Sentiment Distribution")
                    plt.tight_layout()
                    st.pyplot(chart)
                    plt.close(chart)

                    st.dataframe(results, use_container_width=True)

                    st.download_button(
                        "Download analyzed CSV",
                        data=results.to_csv(index=False).encode("utf-8-sig"),
                        file_name="brandpulse_results.csv",
                        mime="text/csv"
                    )

    except Exception as error:
        st.error("Could not process the CSV.")
        st.code(str(error))


#foot
st.markdown("---")
st.caption(
    "BrandPulse AI · Predictions are estimates, not guaranteed facts."
)
