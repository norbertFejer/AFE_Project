import settings as stt
import constants as const
import trainModel
import evaluateModel

from numpy.random import seed
seed(const.RANDOM_STATE)
from tensorflow import set_random_seed
set_random_seed(const.RANDOM_STATE)


def main():

    if stt.sel_method == stt.Method.TRAIN:
        tm = trainModel.TrainModel()
        tm.train_model()

    if stt.sel_method == stt.Method.EVALUATE:
        em = evaluateModel.EvaluateModel()
        em.evaluate_model()

main()