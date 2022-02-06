# Goal:
The project goal is to create a machine learning pipeline to classify disaster events from a dataset provided by Figure Eight containing real messages. The final outcome is a web app where an emergency worker can enter a new message and get classification results in different categories.

# Files:
- process_data.py: This code extracts data from both CSV files: messages.csv (containing message data) and categories.csv (classes of messages) and creates an SQLite database containing a merged and cleaned version of this data.
- train_classifier.py: This code takes the SQLite database produced by process_data.py as an input and uses the data contained within it to train and tune a ML model for categorizing messages. The output is a pickle file containing the fitted model. Test evaluation metrics are also printed as part of the training process.
- disaster_messages.csv, disaster_categories.csv contain sample messages (real messages that were sent during disaster events) and categories datasets in csv format.
- run.py: for interactive classification based on trained model of arbitary text data and visualisations of some data

# Installation:
- Python3
- Machine Learning Libraries: NumPy, Pandas, Scikit-Learn
- Natural Language Process Libraries: nltk
- SQLlite Database Libraries: SQLalchemy
- Model Loading and Saving Library: Pickle
- Web App and Data Visualization: Flask, Plotly

# Instructions for execution:
- Run the following commands in the project's root directory to set up your database and model.

- To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response.db
- To run ML pipeline that trains classifier and saves python models/train_classifier.py data/disaster_response.db models/classifier.pkl
- Run the following command in the app's directory to run your web app. python run.py

- Go to http://0.0.0.0:3001/ to see the web app.

# Acknowledgments:
This project has been completed as part of the Data Science Nanodegree on Udacity. The data was collected by Figure Eight and provided by Udacity.
