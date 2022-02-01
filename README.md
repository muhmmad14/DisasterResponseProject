Goal:
The project goal is to create a machine learning pipeline to classify disaster events from a dataset provided by Figure Eight containing real messages. The final outcome is a web app where an emergency worker can enter a new message and get classification results in different categories.

Installation:
- Python3
- Machine Learning Libraries: NumPy, Pandas, Scikit-Learn
- Natural Language Process Libraries: nltk
- SQLlite Database Libraries: SQLalchemy
- Model Loading and Saving Library: Pickle
- Web App and Data Visualization: Flask, Plotly

Instructions for execution:
- Run the following commands in the project's root directory to set up your database and model.

- To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster_response.db
- To run ML pipeline that trains classifier and saves python models/train_classifier.py data/disaster_response.db models/classifier.pkl
- Run the following command in the app's directory to run your web app. python run.py

- Go to http://0.0.0.0:3001/ to see the web app.
