import os

def make_dir(directory):
	if not os.path.exists(directory):
		os.makedirs(directory)

# traverse all the subdirectories
def list_dir(directory):
	paths = []
	for root, dirs, files in os.walk(directory):
		for subdir in dirs:
			paths.append(os.path.join(root.replace(directory,''), subdir))
	return paths

def list_dir_single(directory):
	for root, dirs, files in os.walk(directory):
		return dirs

def list_files_single(directory):
	for root, dirs, files in os.walk(directory):
		return files