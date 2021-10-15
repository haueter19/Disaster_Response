import json
import plotly
import pandas as pd
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import *
#from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """
    INPUT: text string
    OUTPUT: cleaned, tokenized, lemmatized list in lowercase with punctuation and stop words removed
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        if re.match(r'[^\w]', clean_token)==None:
            if clean_token not in stop_words:
                clean_tokens.append(clean_token)
            else:
                pass
        else:
            pass

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/Disaster.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    viz = df.copy()
    viz['token_count'] = viz['message'].apply(lambda x: len(tokenize(x)))
    viz['bin'] = pd.cut(viz.token_count, bins=[0, 10, 25, 50, 100, 10000], labels=['<10', '10-24', '25-49', '50-99', '100+'])

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=viz.columns[4:-2],
                    y=viz.iloc[:,4:-2].sum().sort_values(ascending=False)
                )
            ],

            'layout': {
                'title': 'Count of Messages by Type',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Message Type"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=viz.groupby('bin')['message'].count().index,
                    y=viz.groupby('bin')['message'].count().values
                )
            ],

            'layout': {
                'title': 'Message Token Length by Bin',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Bin Range"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
