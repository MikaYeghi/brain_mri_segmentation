ROOT_PATH = "/home/yemika/Mikael/Code/Python/Kaggle/brain_mri_segmentation"

# Training parameters
BATCH_SIZE = 32
LR = 0.01
SCHEDULER_GAMMA = 0.5
N_EPOCHS = 1
VAL_FREQ = 1
SAVE_PATH = "saved_models/"
LOAD_MODEL = False              # For training only. Inference automatically loads SAVE_PATH/MODEL_NAME

# Inference parameters
SAVE_PREDS = True
PREDS_PATH = "predictions/"
ROUNDED_SAVE = True

# Common parameters
MODEL_NAME = "model.pt"