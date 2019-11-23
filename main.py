import settings as stt
import trainModel


def main():

    if stt.sel_method == stt.Method.TRAIN:
        tm = trainModel.TrainModel()
        tm.train_model()

main()