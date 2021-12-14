from flask import Flask, request, render_template
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
import redis

nltk.download('stopwords')

set(stopwords.words('english'))

app = Flask(__name__)
cache = redis.Redis(host='redis', port=6379)

@app.route('/')
def my_form():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    stop_words = stopwords.words('english')
    
    #convert to lowercase
    text1 = request.form['text1'].lower()
    
    text_final = ''.join(c for c in text1 if not c.isdigit())

    #remove stopwords    
    processed_doc1 = ' '.join([word for word in text_final.split() if word not in stop_words])

    sid_obj = SentimentIntensityAnalyzer()
    sentiment_dict = sid_obj.polarity_scores(text=processed_doc1)
    compound = round((1 + sentiment_dict['compound'])/2, 2)
    
    res = "Neutral"
    if (compound > 0.5) :
        res = "Positive "
    elif (compound < 0.5) :
        res = "Negative "
    else : 
        res = "Neutral "

    return render_template('form.html', final=compound, text1=text_final,text2=sentiment_dict['pos'],text5=sentiment_dict['neg'],text4=compound,text3=sentiment_dict['neu'], text6=res)


