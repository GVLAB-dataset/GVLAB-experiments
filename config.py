import os

TEST = 'test'
DEV = 'dev'
TRAIN = 'train'

SPLIT_PATH = os.path.join(os.path.dirname(__file__), 'assets/splits')
SWOW_DATA_PATH = os.path.join(os.path.dirname(__file__), 'assets/gvlab_swow_split.csv')
SWOW_SPLIT_PATH = os.path.join(os.path.dirname(__file__), 'assets/swow_split.json')
IMAGES_FOLDER_PATH = os.path.join(os.path.dirname(__file__), 'assets/images')
# IMAGES_FOLDER_PATH = '/Users/yonatab/data/image_associations/gvlab-bucket'
MODEL_RESULTS_PATH = os.path.join(os.path.dirname(__file__), 'models_results')
TRAIN_RESULTS_PATH = os.path.join(os.path.dirname(__file__), 'models_results/train')
