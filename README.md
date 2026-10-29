# BrandPulse AI – Twitter Sentiment Analysis

## About the Project 

BrandPulse AI is a machine learning project that analyzes the sentiment of tweets and classifies them as positive, negative, or neutral.

The project uses airline-related tweets to train and test two different approaches to sentiment analysis: Logistic Regression with TF-IDF and a deep learning model based on LSTM. I also built a Streamlit web app where users can enter a tweet, upload a CSV file, and view sentiment analysis results.

The main aim of this project was to understand how classical machine learning and deep learning approaches work with text data and compare their performance on the same task.

## Features

- Predicts positive, negative, and neutral sentiment from text.
- Uses TF-IDF and Logistic Regression for classical machine learning.
- Uses an LSTM neural network for deep learning.
- Allows users to analyze individual tweets.
- Supports bulk sentiment analysis through CSV uploads.
- Displays sentiment distribution and trends in the dashboard.
- Allows analyzed CSV results to be downloaded.
- Includes notebooks for preprocessing, model training, and testing.

## Models and Results

Two models were developed and evaluated for this project.

| Model               | Approach                        | Test Accuracy |
| ------------------- | ------------------------------- | ------------: |
| Logistic Regression | TF-IDF + Logistic Regression    |         80.2% |
| LSTM                | Embedding + LSTM neural network |        79.41% |

The Logistic Regression model achieved slightly higher overall accuracy on the test set. The LSTM model provided a deep learning comparison. Since the dataset contains more negative tweets than positive and neutral tweets, accuracy alone does not tell the whole story; class-wise precision, recall, and F1-score are also useful for evaluating the models.

These results are from the experiments conducted during development and may vary if the data split, preprocessing, or model settings change.

## Dataset

The project uses an airline tweet dataset containing tweet text and sentiment labels.

The three sentiment classes are:

- Positive
- Negative
- Neutral

The text is cleaned before training so that unnecessary URLs, mentions, and extra whitespace do not interfere with the analysis.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- TensorFlow / Keras
- Joblib
- Streamlit
- Matplotlib / Plotly, where used for visualizations
- Jupyter Notebook

## Project Structure

```text
Brandpulse_AI
├── data/
│   └── Tweets.csv
├── models/
│   ├── logistic_regression.pkl
│   ├── lstm_label_encoder.pkl
│   ├── lstm_sentiment_model.keras
│   ├── lstm_tokenizer.pkl
│   └── tfidf_vectorizer.pkl
├── notebook/
│   ├── 01_Preprocessing_Classical.ipynb
│   ├── 02_Deep_Learning_LSTM.ipynb
│   └── 03_Test_LSTM.ipynb
├── reports/
├── .gitignore
├── app.py
├── check_models.py
├── README.md
├── requirements.txt
└── test.csv
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Vashuchoudhary001/BrandPulse-AI.git
```

### 2. Open the project folder

```bash
cd BrandPulse-AI
```

### 3. Create a virtual environment (recommended)

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install the dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL, usually `http://localhost:8501`, which you can open in your browser.

## How to Use

1. Start the Streamlit application.
2. Enter a tweet to predict its sentiment, or use the CSV upload option for multiple tweets.
3. Select the available model and review the predictions.
4. Explore the sentiment distribution and trend charts.
5. Download the analyzed results when needed.

## Future Improvements

- Improve the handling of neutral and ambiguous tweets.
- Test the models on more recent tweets and different datasets.
- Add more evaluation metrics and model comparisons.

## Author

Developed as a machine learning and NLP project to explore sentiment analysis using classical machine learning, deep learning, and Streamlit.
