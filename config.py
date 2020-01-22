import os

PATH = os.path.dirname(os.path.abspath(__file__))

DATASETS_DIR = os.path.join(PATH, 'datasets')

OUTPUT_DIR = os.path.join(PATH, 'output')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
PARAMS_DIR = os.path.join(OUTPUT_DIR, 'params')
VIDEOS_DIR = os.path.join(OUTPUT_DIR, 'videos')
TENSORBOARD_DIR = os.path.join(OUTPUT_DIR, 'tensorboard')
TQDM_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'

LOGGING_FORMAT = '[%(levelname)s] %(module)s:%(message)s'
