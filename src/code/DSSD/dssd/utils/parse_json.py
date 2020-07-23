import json

def load_json(f):
  with open(f, 'r') as fp:
    return json.load(fp)


def save_json(obj, f, *args, **kwargs):
  with open(f, 'w') as fp:
    json.dump(obj, fp, *args, **kwargs)