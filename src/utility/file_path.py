import os.path

DIRECTORY_PATH = os.getcwd()
REPO_PATH = DIRECTORY_PATH.split(os.sep)
REPO_PATH = os.sep.join(REPO_PATH[:-2])

LOG_PATH = os.path.join(REPO_PATH, 'log')
OBJECT_PATH = os.path.join(REPO_PATH, 'objectset')
DATASET_PATH = os.path.join(REPO_PATH, 'dataset')
INPUT_PATH = os.path.join(REPO_PATH, 'inputset')
RESULT_PATH = os.path.join(REPO_PATH, 'result')
MODEL_PATH = os.path.join(REPO_PATH, 'model')
FEATURE_PATH = os.path.join(DIRECTORY_PATH, 'feature_builder')
