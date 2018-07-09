import tensorflow as tf
from PIL import Image
import numpy as np
from os import listdir
import os
import uuid

def get_images(dir, type):
    files = listdir(dir)
    if len(files) > 0:
        im_arr = np.array(Image.open(dir + '\\' + files[0]).getdata())
        if type == 'rap':
            lab_arr = np.array([1, 0])
        else:
            lab_arr = np.array([0, 1])

        for f in files[1:]:
            image = np.array(Image.open(dir + '\\' + f).getdata())
            im_arr = np.vstack([im_arr, np.atleast_2d(image)])
            if type == 'rap':
                lab_arr = np.vstack([lab_arr, np.array([[1, 0]])])
            else:
                lab_arr = np.vstack([lab_arr, np.array([[0, 1]])])

        return im_arr, lab_arr
    return [],[]


def png_input_fn(rap_dir,classical_dir):
    rap_im_arr,rap_lab_arr = get_images(rap_dir, 'rap')
    classical_im_arr, classical_lab_arr = get_images(classical_dir, 'classical')
    if rap_im_arr != []:

        im_arr = np.vstack([rap_im_arr,classical_im_arr])
        lab_arr = np.vstack([rap_lab_arr, classical_lab_arr])

        return im_arr, lab_arr
    return [],[]


def get_model(input_shape):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(300,activation="sigmoid",input_shape=input_shape))
    model.add(tf.keras.layers.Dense(2,activation="sigmoid"))

    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam())

    return model


# trains the model on the same input data 'train_runs' times. Beware of overfitting!
def train_model(input, model, train_runs):
    for x in range(train_runs):
        model.fit(x=input[0], y=input[1], batch_size=1)


def evaluate_predictions(eval_input, model, num_files):
    predictions = model.predict(x=eval_input[0], batch_size=1)

    c = 0
    for x in range(num_files):
        if eval_input[1][x][0] == 1:
            if predictions[x][0] > .5:
                c = c + 1
        else:
            if predictions[x][0] < .5:
                c = c + 1

    return c/num_files

rap_dir = '..\\train_files\\rappng'
classical_dir = '..\\train_files\\classicalpng'
rap_eval_dir = '..\\eval_files\\rappngeval'
classical_eval_dir = '..\\eval_files\\classicalpngeval'

runs = 1
accum_for_avg = 0
# this loop gets the model and trains then evaluates the predictions 'runs' times
for x in range(runs):
    input = png_input_fn(rap_dir=rap_dir,classical_dir=classical_dir)
    if input[0] != []:
        model = get_model((128*128,))
        train_runs = 5

        train_model(input, model, train_runs)

        rap_eval_png_list = os.listdir(rap_eval_dir)
        classical_eval_png_list = os.listdir(classical_eval_dir)
        num_eval_files = len(rap_eval_png_list) + len(classical_eval_png_list)
        eval_input = png_input_fn(rap_dir=rap_eval_dir,classical_dir=classical_eval_dir)
        if (eval_input[0] != []):
            predictions = model.predict(x=eval_input[0],batch_size=1)

            percentage = evaluate_predictions(eval_input, model, num_eval_files)

            print(percentage)
            accum_for_avg = accum_for_avg + percentage

        model.save('../models/model_' + str(uuid.uuid4()) + '.h5')

# print average prediction accuracy for all of the training & prediction runs
print(accum_for_avg/runs)


