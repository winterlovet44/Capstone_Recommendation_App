import pickle


def ensure_type_path(path, typefile='csv'):
    if path.split(".")[-1] != typefile:
        path += "." + typefile
    return path
#
#
# def save_model(model, path):
#     with open(path, "wb") as f:
#         pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
#         f.close()
#     return


def save_model(model, path):
    if hasattr(model, "save"):
        model.save(path)
    else:
        with open(path, "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()
    return


def load_model(path):
    with open(path, "rb") as f:
        model = pickle.load(f)
        f.close()
    return model
