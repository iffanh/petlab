import json

def read_json(jsonfilename):
    with open(jsonfilename, 'r') as j:
        jsondata = json.loads(j.read())
        
    return jsondata

def save_to_json(filename, d):
    with open(filename, 'w') as f:
        json.dump(d, f, indent=4)