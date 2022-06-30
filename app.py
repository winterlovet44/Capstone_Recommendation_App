import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
from sqlalchemy import create_engine
from utils.util import load_model

from utils.app_utils import load_data, get_res_result


app = Flask(__name__)


# load data
engine = create_engine("sqlite:///metadata")
ds = load_data()

# load model
als = load_model("../models/als.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    # Find distribution of categories
    features = df.columns[4:]
    data = df[features].T.sum(axis=1).to_dict()
    # Find distribution of categories of each item
    data_count = df[features].sum(axis=1).value_counts()
    x_ = data_count.index.tolist()
    y_ = data_count.values.tolist()

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=list(data.keys()),
                    y=list(data.values())
                )
            ],

            'layout': {
                'title': 'Distribution of Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=x_,
                    y=y_
                )
            ],

            'layout': {
                'title': 'Amount of Categories by item',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Total Categories"
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
    result = get_res_result(model=als, dataset=ds, user_id=query)

    # This will render the go.html Please see that file.
    return render_template(
        'recommend.html',
        tables=[result.to_html(classes='data')], titles=result.columns.values
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()