import settings as stt
import trainModel
import evaluateModel


def main():

    if stt.sel_method == stt.Method.TRAIN:
        tm = trainModel.TrainModel()
        tm.train_model()

    if stt.sel_method == stt.Method.EVALUATE:
        em = evaluateModel.EvaluateModel()
        em.evaluate_model()

main()