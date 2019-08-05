import settings
from keras.models import load_model
import constants as const
import dataset
from keras.utils import to_categorical


def evaluate_model(testX, testy):

    modelName = const.BASE_PATH + '/' + const.TRAINED_MODELS_PATH + '/' + str(settings.selectedModel) + '_' + const.SESSION_NAME + '_trained.h5'
    model = load_model(modelName)

    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=const.BATCH_SIZE, verbose=1)

    score = accuracy * 100.0
    print('>#%d: %.3f' % (1, score))


def run_experiment():

    print('Loading test dataset...')
    testX, testy = dataset.create_test_dataset(const.SESSION_NAME)

    testy = to_categorical(testy)
    print('Loading test dataset finished')
    print(testX.size)

    print(testX.shape)

    evaluate_model(testX, testy)


run_experiment()