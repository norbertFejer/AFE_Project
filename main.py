import settings as stt
import constants as const
import trainModel
import evaluateModel

# Setting random states to get reproducible results
from numpy.random import seed
seed(const.RANDOM_STATE)
from tensorflow import set_random_seed
set_random_seed(const.RANDOM_STATE)
#tf.random.set_seed(const.RANDOM_STATE)


if __name__ == "__main__":

    if stt.sel_method == stt.Method.TRAIN or stt.sel_method == stt.Method.TRANSFER_LEARNING:
        tm = trainModel.TrainModel()
        tm.train_model()
            
    if stt.sel_method == stt.Method.EVALUATE:
        em = evaluateModel.EvaluateModel()
        em.evaluate_model()
        