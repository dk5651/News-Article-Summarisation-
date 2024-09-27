from flask import Flask, request, render_template
from transformers import pipeline
import requests

app = Flask(__name__)

# Use a summarization model that is generally reliable
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)

API_KEY = '5a4ce35ea3274427870b59206dfa0cca'  # Replace with your actual API key

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/fetch_articles', methods=['POST'])
def fetch_articles():
    topic = request.form['topic']
    url = f'https://newsapi.org/v2/everything?q={topic}&apiKey={API_KEY}'

    response = requests.get(url)

    if response.status_code == 200:
        articles = response.json().get('articles', [])
    else:
        articles = []

    return render_template('index.html', articles=articles)

@app.route('/summarize', methods=['POST'])
def summarize():
    text = request.form['text']
    if not text:
        return render_template('index.html', summary="No text to summarize.", original="")
    
    try:
        summary = summarizer(
            text,
            max_length=130,
            min_length=30,
            num_beams=4,
            do_sample=False,
            early_stopping=True
        )
        return render_template('index.html', summary=summary[0]['summary_text'], original=text)
    except Exception as e:
        print(f"Error during summarization: {e}")
        return render_template('index.html', summary="Error summarizing article.", original=text)

if __name__ == '__main__':
    app.run(debug=True)
