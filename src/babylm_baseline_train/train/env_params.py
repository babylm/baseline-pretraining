import os
from pt_framework.dist_utils import use_tpu
from ..env_vars import ROOT_DIR, ROOT_DIR_FREQ


MODEL_SAVE_FOLDER = os.environ.get(
        'BABYLM_MODEL_SAVE_FOLDER',
        os.path.join(
            ROOT_DIR, 'models/'))
REC_SAVE_FOLDER = os.environ.get(
        'BABYLM_REC_SAVE_FOLDER',
        os.path.join(
            ROOT_DIR_FREQ, 'model_recs/'))
USE_TPU = use_tpu()
MANUAL_FORCE_TPU = int(os.environ.get('MANUAL_FORCE_TPU', 0)) == 1
USE_TPU = USE_TPU or MANUAL_FORCE_TPU
