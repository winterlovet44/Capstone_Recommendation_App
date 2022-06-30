import click
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sqlalchemy import create_engine
from utils.util import ensure_type_path


DBNAME = "data/metadata"
FILEPATH = "data/metadata.json"


def load_meta(path):
    meta = pd.read_csv(path,
                       delimiter="::",
                       encoding="ISO-8859-1",
                       header=None,
                       names=["MovieID", "Title", "Genres"],
                       engine='python'
                       )
    return meta


def nested_category_encode(data):
    mlb = MultiLabelBinarizer()
    res = pd.DataFrame(mlb.fit_transform(data.Genres),
                       columns=mlb.classes_,
                       index=data.index)


# def ensure_csv_path(path):
#     if path.split(".")[-1] != "csv":
#         path += "meta.csv"
#     return path


@click.command()
@click.option('--input_path', default="raw/movies.dat", help='Filepath contains movielens 1M dataset')
@click.option('--filename', default=FILEPATH, help='Output path to write cleaned metadata')
@click.option('--tablename', default=None, help='Output path to write cleaned metadata')
def run(input_path, filename, tablename):
    print("Load raw data...\n")
    meta = load_meta(input_path)
    print("Extract release from title of movie...\n")
    meta['release_year'] = meta.Title.str.extract("(\d+)")
    print("Remove year has been extracted from title\n")
    meta.Title = meta.Title.apply(lambda x: x[:-7])
    print("Replace '|' in Genres to ','\n")
    meta.Genres = meta.Genres.apply(lambda x: x.replace("|", ","))
    # genre_encode_df = nested_category_encode(meta.Genres)
    # new_meta = pd.concat([meta.drop(columns=['Genres']), genre_transform], axis=1)
    if filename:
        write_to = ensure_type_path(filename, "json")
        print(f"Write metadata to {filename}\n")
        meta.to_json(write_to, orient='records')
    if tablename:
        engine = create_engine(f"sqlite:///{DBNAME}")
        print(f"Write metadata to database file: {DBNAME}")
        print(f"With table name: {tablename}\n")
        meta.to_sql(name=tablename, con=engine, if_exists="replace")
    print("Done!!!")
    return


if __name__ == '__main__':
    run()
