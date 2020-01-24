# Automatic Feature Extraction

## Mouse Dynamics based User Recognition using Deep Learning

Behavioural biometrics provides an extra layer of security for user authentication mechanisms. Among behavioural biometrics, mouse dynamics provides a non-intrusive layer of security. In this project we propose novel convolutional neural networks for extracting the features from the time series of users’ mouse movements.

## Installation

For creating a virtual environment we used [Anacoda](https://docs.anaconda.com/anaconda/install/), so you have to install it for the purpose of reproductibility.

The environment.yml file contains all the necessary dependencies for the project.

Creating the environment from yml file:
```bash
conda env create -f environment.yml
```
Activating the environmet for usage:
```bash
conda activate tensorflow_cpu
```
## Initial Configuration

After downloading the two separate datasets ([Balabit](https://github.com/balabit/Mouse-Dynamics-Challenge) and [DFL](https://ms.sapientia.ro/~manyi/DFL.html)), you need to specify these locations in the constants.py file.

## Building

Using the Anaconda Prompt you simply run the command:
```bash
python ./main.py
```

## Features

You can use a model from given neural network models, see their architectures, manage training with different methods and evaluate these results. Of course you can plot these data, make a result file as .csv format or simply print the given results to the screen. 

## Configuration

### Define important constanst in constants.py

#### Define block size of given samples

`BLOCK_SIZE = 128` gives the number of consecutive mouse movements (1 movement represents one row in the train.csv file).

#### Define user name

`USER_NAME = 'user35'` defines which user's data we want to use during measurement.

#### Define model for transfer learning 

`USED_MODEL_FOR_TRANSFER_LEARNING = 'model.h5'` defines the model name in case of transfer learning.

#### Define train - test split value

`TRAIN_TEST_SPLIT_VALUE = 0.2` defines that we want to use 20% for testing and 80% of data for training.

## Training the models

* First of all you have to set `sel_method = Method.TRAIN` or `sel_method = Method.TRANSFER_LEARNING` in settings.py.

* Setting `BLOCK_NUM` you define how many data you want to use during the training (For example `BLOCK_NUM = 50` you use 50 * BLOCK_SIZE rows from the training files).

* With `sel_model` you choose one of the two neural model architecture (**CNN** and **Time Distributed**).

* To defining the type of samples negative/positive balance rate you use `sel_balance_type`.

* There are two main type of training: **identification** and **authentication**, you set this with `sel_user_recognition_type`.

* To dealing with the chunk data you use *drop chunks* or *concatenate chunks* in `sel_chunck_samples_handler` parameter.

* With `sel_train_user_number` you specify that you want to traint the model for all user, or you can pick a single one from the dataset.

## Evaluating the models

* First of all you have to set `sel_method = Method.EVALUATE` in settings.py.

* You can specify that the given dataset contains separate train and test samples using the `sel_dataset_type` parameter.

* With the `sel_evaluation_metrics` you can set the evaluation metrics from the predefined ones.

* The `sel_evaluation_type` specifies that you want to perform **action based** or **session based** evaluation.

## Authors

**Margit Antal** - Sapientia Hungarian University of Transylvania Department of Mathematics– Informatics, Tirgu Mures
**Norbert Fejér** - Sapientia Hungarian University of Transylvania Department of Electrical Engineering, Tirgu Mures

## Acknowledgements

The work was supported by Accenture Industrial Software Solutions.