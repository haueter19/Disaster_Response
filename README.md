# Disaster_Response
#### Udacity Data Science Nanodegree Program

### Purpose
To build a pipeline that trains a model from labeled data and situates in an app for direct use

### Prereqs
<ul>
  <li>Python 3.8.2</li>
  <li>Pandas</li>
  <li>Plotly</li>
  <li>Flask</li>
  <li>NLTK</li>
  <li>SQL Alchemy</li>
  <li>Scikit Learn</li>
 </ul>

### How To
<ul>
  <li>Install the requisite packages</li>
  <li>From command line, navigate to app folder then type <code>python run.py</code></li>
  <li>Open a browser to http://localhost:3001 or http://0.0.0.0:3001</li>
</ul>

### Structure
```
project
|  app
|  |  run.py
|  |  templates
|  |  |  go.html
|  |  |  master.html
|  data
|  |  disaster_categories.csv (labeled categories)
|  |  disaster_messages.csv (message text and genre)
|  |  process_data.py
|  models
|  |  classifier.pkl
|  |  train_classifier.pkl
```

