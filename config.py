import os

PATH = os.path.dirname(os.path.abspath(__file__))

DATASETS_DIR = os.path.join(PATH, 'datasets')

OUTPUT_DIR = os.path.join(PATH, 'output')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
PARAMS_DIR = os.path.join(OUTPUT_DIR, 'params')
VIDEOS_DIR = os.path.join(OUTPUT_DIR, 'videos')
TENSORBOARD_DIR = os.path.join(OUTPUT_DIR, 'tensorboard')
TQDM_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
IGNITE_BAR_FORMAT = '{desc}[{n_fmt}/{total_fmt}] {percentage:3.0f}%|{bar}|{postfix} [{elapsed}<{remaining}]'
METRICS_REPORT_FORMAT = " - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
EVAL_REPORT_FORMAT = "Validation Results" + METRICS_REPORT_FORMAT
TRAIN_REPORT_FORMAT = "Training Results" + METRICS_REPORT_FORMAT

LOGGING_FORMAT = '[%(levelname)s] %(module)s:%(message)s'
