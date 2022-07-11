# Recommendation Web-app

### Table of Contents

1. [Project Description](#motivation)
2. [Dependencies](#dependencies)
3. [Installation](#installation)
4. [File Descriptions](#files)
5. [Instructions](#results)



## Project Description<a name="motivation"></a>

In this project, I build a application about movie recommender system with Movielens 1M dataset.
For more information about the dataset, please check here.
I use 
I also implement a simple web-app for perform recommend for an user or an item.

This project contains four steps of ML pipeline:

1. ETL: Clean data and save cleaned data to file and database.
2. Feature engineering: Transform feature to meet model fitting.
3. Modelling: Build a Machine learning pipeline to feature engineering and train ML model.
4. Model serving: Build Flask web app to predict user's input query.



## Dependencies <a name="dependencies"></a>

To run code in this project, your enviroments need:
1. [Pandas](https://pandas.pydata.org/)
2. [Numpy](https://numpy.org/)
3. [Scikit-learn](https://scikit-learn.org/stable/)
4. [implicit](https://github.com/benfred/implicit)
5. [SQLAlchemy](https://sqlalchemy.org/)
6. [Flask](https://flask.palletsprojects.com/)
7. [Click](https://click.palletsprojects.com/en/8.1.x/)


## Installation <a name="installation"></a>

The code was implemented in Python 3.9. All necessary package was contained in `requirements.txt` file.

For quick installation:
```sh
pip install -r requirements.txt
```


## File Descriptions <a name="files"></a>

```bash
├── README.md
├── app
│   ├── run.py # Flask app
│   └── templates # Folder contains html file to render web app
│       ├── go.html
│       └── master.html
├── data
│   ├── DisasterResponse.db # SQLite database file
│   ├── YourDatabaseName.db
│   ├── disaster_categories.csv # Raw data
│   ├── disaster_messages.csv # Raw data
│   └── process_data.py # ETL pipeline
├── models
│   ├── classifier.pkl # Saved model
│   └── train_classifier.py # Model pipeline
└── requirements.txt # Package requirement
```

## Instructions<a name="results"></a>

1. ETL pipeline

We need to pre-processing for user's dataset and item's dataset.
To run ETL pipeline for clean user dataset, run the code below:

```bash
cd data
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
```

To run ETL pipeline for clean item's dataset:

```bash
cd data
python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
```

2. To build and train model

```bash
cd models
python train_classifier.py ../data/DisasterResponse.db classifier.pkl
```

3. Run web app

Run the code below to start the web app at localhost

```bash
cd app
python run.py
```

And go to [http://localhost:3000](http://localhost:3000/) to see the web app


