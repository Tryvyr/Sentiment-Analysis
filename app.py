import re
from flask import Flask, render_template, request
import pandas as pd
import pickle
import base64
from io import BytesIO
from wordcloud import WordCloud
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return 'No file uploaded'
        file = request.files['file']

        if file.filename.endswith('.csv'):
            # read the CSV file using pandas
            try:
                df = pd.read_csv(file)
            except ValueError:
                return 'CSV file format cannot be determined'
        else:
            return 'File must be an Excel or CSV file'
        
        # Open Model
        mnb_model = model()
        vectorizer_model = vectorizer()

        # Clean
        df['review'] = df['review'].apply(preprocess_text)

        # Predict Sentiment using the model
        df = user_dataset(df, mnb_model, vectorizer_model)

        # Show Common Words with Stop Words
        words_freq_pos = words_freq_method(df, 'positive')
        words_freq_neg = words_freq_method(df, 'negative')

        # World Cloud
        img_str_pos = wordcloud(words_freq_pos)
        img_str_neg = wordcloud(words_freq_neg)

        # Count sentiment
        pos, neg, total = count_sentiment(df)

        # Dataframe to Html
        summary = df.to_html()

        return render_template('results.html', summary=summary, positive_count=pos, negative_count=neg, sum=total, img_str_pos=img_str_pos, img_str_neg=img_str_neg, link=request.form['link'], name=request.form['name'])
    return render_template('upload.html')


@app.route('/predict', methods=['POST'])
def predict_sentiment():

    # Open Model
    mnb_model = model()
    vectorizer_model = vectorizer()


    # Get Input
    sentence = request.form['sentence']

    # Clean
    sentence = preprocess_text(sentence)

    # Transform the sentence into numerical features using the same vectorizer object
    X_sentence = vectorizer_model.transform([sentence])

    # Use the trained MultinomialNB model to predict the sentiment of the sentence
    sentiment = mnb_model.predict(X_sentence)

    return  render_template('upload.html' , sentiment=sentiment)

def model():
    with open('C:/Users/Maike/OneDrive/Desktop/Thesis/App/modelsv1/mnb_model.pkl', 'rb') as f:
        mnb_model = pickle.load(f)
    return mnb_model

def vectorizer():
    with open('C:/Users/Maike/OneDrive/Desktop/Thesis/App/modelsv1/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer


def words_freq_method(df, sentiment):
    vectorizer_model = vectorizer()
    df_sentiment = df[df['sentiment'] == sentiment]
    X_train_counts = vectorizer_model.fit_transform(df_sentiment['review'])
    word_counts = X_train_counts.sum(axis=0)
    words_freq = [(word, word_counts[0, idx]) for word, idx in vectorizer_model.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: -x[1])
    return words_freq

def wordcloud(words_freq):
    wordcloud = WordCloud(width=800, height=800, background_color='white').generate_from_frequencies(dict(words_freq[:50]))
    img_file = BytesIO()
    plt.figure(figsize=(6,6), facecolor=None)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(img_file, format='png')
    img_file.seek(0)
    img_str = base64.b64encode(img_file.read()).decode('ascii')
    return img_str

def user_dataset(df, mnb_model, vectorizer_model):
    predictions = []

    for i, word in enumerate(df['review']):
        # Transform the word using the vectorizer
        X_word = vectorizer_model.transform([word])
        
        # Make a prediction using the Naive Bayes model
        y_word = mnb_model.predict(X_word)
        
        # Append the prediction to the list of predictions
        predictions.append(y_word[0])

    # Store the predictions in a new column in the DataFrame
    df['sentiment'] = predictions
    return df

def count_sentiment(df):
    sentiment_counts = df['sentiment'].value_counts()
    positive_count = sentiment_counts['positive']
    negative_count = sentiment_counts['negative']
    sum = positive_count+negative_count
    return positive_count, negative_count, sum

def preprocess_text(review):
    vectorizer1 = vectorizer()
    if pd.isna(review):
        review = ''
    review = review.lower()
    review = re.sub(r'[^a-zA-Z0-9\s]', '', review)
    review = re.sub(r'\s+', ' ', review)
    review = re.sub(r'\d+', '', review)
    words = review.split()
    review = [word for word in words if word not in vectorizer1.get_stop_words()]
    return ' '.join(review)
if __name__ == '__main__':
    app.run(debug=True)