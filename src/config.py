"""
Configuration Files
"""

DATASET = "CIFAR_10_10%"
DATASET_CLASS_LIST = ['Cat', 'Dog', 'Airplane', 'Bird', 'Frog', 'Horse', 'Deer', 'Automobile', 'Ship', 'Truck']
COMPLETE_DATASET_DIR = "Data/" + DATASET
NEW_DATASET_DIR = "Data/New_" + DATASET
STARTING = 2
TRAIN = "/train/"
TEST = "/test/"
CLASSES_TO_ADD = 2
BATCH_SIZE = 128
PATH_TO_SAVE_MODEL = "Models/model_weights.pth"

TOTAL_EPOCHS = 3
