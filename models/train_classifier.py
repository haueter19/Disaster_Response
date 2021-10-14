import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(database_filepath):
    #engine = create_engine(database_filepath)
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages', engine)
    category_names = df.columns[4:]
    X = df.iloc[:,1]
    Y = df.iloc[:,4:]
    Y['related'] = Y.related.apply(lambda x: 1 if x==2 else x)
    return X, Y, category_names


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline(
        [
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=5,min_samples_leaf=1,max_features='auto',n_jobs=-1)))
        ]
    )
    parameters = {
        'clf__estimator__n_estimators':[20, 70],
        #'clf__estimator__min_samples_leaf':[2, 5, 7],
        #'clf__estimator__max_features': [0.5, 1, "log2"]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv

def evaluate_model(model, X_test, y_test, category_names):
    y_pred = model.predict(X_test)
    return print(classification_report(y_test, y_pred,target_names=category_names))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as model_file:
        pickle.dump(model, model_file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
