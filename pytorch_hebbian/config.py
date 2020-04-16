import os

PATH = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(PATH, '..')

DATASETS_DIR = os.path.join(ROOT, 'datasets')
OUTPUT_DIR = os.path.join(ROOT, 'output')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
PARAMS_DIR = os.path.join(OUTPUT_DIR, 'params')
VIDEOS_DIR = os.path.join(OUTPUT_DIR, 'videos')
TENSORBOARD_DIR = os.path.join(OUTPUT_DIR, 'tensorboard', 'runs')

IGNITE_BAR_FORMAT = '{desc}[{n_fmt}/{total_fmt}] {percentage:3.0f}%|{bar}|{postfix} [{elapsed}<{remaining}]'
LOGGING_FORMAT = '[%(levelname)s] %(name)s:%(message)s'
