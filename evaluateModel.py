import constants as const
import settings as stt
import dataset

from keras.models import load_model
from keras.utils import to_categorical
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score

class EvaluateModel:


    