ROOT_PATH = "/home/yemika/Mikael/Code/Python/Kaggle/brain_mri_segmentation"

# Training parameters
N_EPOCHS = 15
BATCH_SIZE = 16
LR = 0.0001
SCHEDULER_GAMMA = 0.5
SCHEDULER_STEP = 10
VAL_FREQ = 5
SAVE_PATH = "saved_models/"
LOAD_MODEL = False              # For training only. Inference automatically loads SAVE_PATH/MODEL_NAME
ENCODER = 'vgg11'

# Inference parameters
SAVE_PREDS = True
PREDS_PATH = "predictions/"
ROUNDED_SAVE = True

# Common parameters
MODEL_NAME = "model.pt"