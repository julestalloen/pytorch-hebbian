import os

PATH = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(PATH, '..')

DATASETS_DIR = os.path.join(ROOT, 'datasets')

OUTPUT_DIR = os.path.join(ROOT, 'output')
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
PARAMS_DIR = os.path.join(OUTPUT_DIR, 'params')
VIDEOS_DIR = os.path.join(OUTPUT_DIR, 'videos')
TENSORBOARD_DIR = os.path.join(OUTPUT_DIR, 'tensorboard', 'runs')
TQDM_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
IGNITE_BAR_FORMAT = '{desc}[{n_fmt}/{total_fmt}] {percentage:3.0f}%|{bar}|{postfix} [{elapsed}<{remaining}]'
METRICS_REPORT_FORMAT = " epoch {}: Avg accuracy: {:.3f}, Avg loss: {:.4f}"
EVAL_REPORT_FORMAT = "Validation" + METRICS_REPORT_FORMAT
TRAIN_REPORT_FORMAT = "Training" + METRICS_REPORT_FORMAT

LOGGING_FORMAT = '[%(levelname)s] %(module)s:%(message)s'
