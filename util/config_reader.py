import json

def config_reader(config_path):
	json_data=open(config_path).read()
	config = json.loads(json_data)
	return config
