import numpy as np

import pandas as pd

import pydicom

import os

import matplotlib.pyplot as plt

import collections

from tqdm import tqdm_notebook as tqdm

from datetime import datetime



from math import ceil, floor, log

import cv2



import tensorflow as tf

import keras



import sys



# from keras_applications.resnet import ResNet50

from keras_applications.inception_v3 import InceptionV3



from sklearn.model_selection import ShuffleSplit



test_images_dir = '../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_test/'

train_images_dir = '../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_train/'
def correct_dcm(dcm):

    x = dcm.pixel_array + 1000

    px_mode = 4096

    x[x>=px_mode] = x[x>=px_mode] - px_mode

    dcm.PixelData = x.tobytes()

    dcm.RescaleIntercept = -1000



def window_image(dcm, window_center, window_width):

    

    if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100):

        correct_dcm(dcm)

    

    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept

    img_min = window_center - window_width // 2

    img_max = window_center + window_width // 2

    img = np.clip(img, img_min, img_max)



    return img



def bsb_window(dcm):

    brain_img = window_image(dcm, 40, 80)

    subdural_img = window_image(dcm, 80, 200)

    soft_img = window_image(dcm, 40, 380)

    

    brain_img = (brain_img - 0) / 80

    subdural_img = (subdural_img - (-20)) / 200

    soft_img = (soft_img - (-150)) / 380

    bsb_img = np.array([brain_img, subdural_img, soft_img]).transpose(1,2,0)



    return bsb_img



# Sanity Check

# Example dicoms: ID_2669954a7, ID_5c8b5d701, ID_52c9913b1



dicom = pydicom.dcmread(train_images_dir + 'ID_5c8b5d701' + '.dcm')

#                                     ID  Label

# 4045566          ID_5c8b5d701_epidural      0

# 4045567  ID_5c8b5d701_intraparenchymal      1

# 4045568  ID_5c8b5d701_intraventricular      0

# 4045569      ID_5c8b5d701_subarachnoid      1

# 4045570          ID_5c8b5d701_subdural      1

# 4045571               ID_5c8b5d701_any      1

plt.imshow(bsb_window(dicom), cmap=plt.cm.bone);

def window_with_correction(dcm, window_center, window_width):

    if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100):

        correct_dcm(dcm)

    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept

    img_min = window_center - window_width // 2

    img_max = window_center + window_width // 2

    img = np.clip(img, img_min, img_max)

    return img



def window_without_correction(dcm, window_center, window_width):

    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept

    img_min = window_center - window_width // 2

    img_max = window_center + window_width // 2

    img = np.clip(img, img_min, img_max)

    return img



def window_testing(img, window):

    brain_img = window(img, 40, 80)

    subdural_img = window(img, 80, 200)

    soft_img = window(img, 40, 380)

    

    brain_img = (brain_img - 0) / 80

    subdural_img = (subdural_img - (-20)) / 200

    soft_img = (soft_img - (-150)) / 380

    bsb_img = np.array([brain_img, subdural_img, soft_img]).transpose(1,2,0)



    return bsb_img



# example of a "bad data point" (i.e. (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100) == True)

dicom = pydicom.dcmread(train_images_dir + "ID_036db39b7" + ".dcm")



fig, ax = plt.subplots(1, 2)



ax[0].imshow(window_testing(dicom, window_without_correction), cmap=plt.cm.bone);

ax[0].set_title("original")

ax[1].imshow(window_testing(dicom, window_with_correction), cmap=plt.cm.bone);

ax[1].set_title("corrected");
def _read(path, desired_size):

    """Will be used in DataGenerator"""

    

    dcm = pydicom.dcmread(path)

    

    try:

        img = bsb_window(dcm)

    except:

        img = np.zeros(desired_size)

    

    

    img = cv2.resize(img, desired_size[:2], interpolation=cv2.INTER_LINEAR)

    

    return img



# Another sanity check 

plt.imshow(

    _read(train_images_dir+'ID_5c8b5d701'+'.dcm', (128, 128)), cmap=plt.cm.bone

);
class DataGenerator(keras.utils.Sequence):



    def __init__(self, list_IDs, labels=None, batch_size=1, img_size=(512, 512, 1), 

                 img_dir=train_images_dir, *args, **kwargs):



        self.list_IDs = list_IDs

        self.labels = labels

        self.batch_size = batch_size

        self.img_size = img_size

        self.img_dir = img_dir

        self.on_epoch_end()



    def __len__(self):

        return int(ceil(len(self.indices) / self.batch_size))



    def __getitem__(self, index):

        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        list_IDs_temp = [self.list_IDs[k] for k in indices]

        

        if self.labels is not None:

            X, Y = self.__data_generation(list_IDs_temp)

            return X, Y

        else:

            X = self.__data_generation(list_IDs_temp)

            return X

        

    def on_epoch_end(self):

        

        

        if self.labels is not None: # for training phase we undersample and shuffle

            # keep probability of any=0 and any=1

            keep_prob = self.labels.iloc[:, 0].map({0: 0.35, 1: 0.5})

            keep = (keep_prob > np.random.rand(len(keep_prob)))

            self.indices = np.arange(len(self.list_IDs))[keep]

            np.random.shuffle(self.indices)

        else:

            self.indices = np.arange(len(self.list_IDs))



    def __data_generation(self, list_IDs_temp):

        X = np.empty((self.batch_size, *self.img_size))

        

        if self.labels is not None: # training phase

            Y = np.empty((self.batch_size, 6), dtype=np.float32)

        

            for i, ID in enumerate(list_IDs_temp):

                X[i,] = _read(self.img_dir+ID+".dcm", self.img_size)

                Y[i,] = self.labels.loc[ID].values

        

            return X, Y

        

        else: # test phase

            for i, ID in enumerate(list_IDs_temp):

                X[i,] = _read(self.img_dir+ID+".dcm", self.img_size)

            

            return X
from keras import backend as K



def weighted_log_loss(y_true, y_pred):

    """

    Can be used as the loss function in model.compile()

    ---------------------------------------------------

    """

    

    class_weights = np.array([2., 1., 1., 1., 1., 1.])

    

    eps = K.epsilon()

    

    y_pred = K.clip(y_pred, eps, 1.0-eps)



    out = -(         y_true  * K.log(      y_pred) * class_weights

            + (1.0 - y_true) * K.log(1.0 - y_pred) * class_weights)

    

    return K.mean(out, axis=-1)





def _normalized_weighted_average(arr, weights=None):

    """

    A simple Keras implementation that mimics that of 

    numpy.average(), specifically for this competition

    """

    

    if weights is not None:

        scl = K.sum(weights)

        weights = K.expand_dims(weights, axis=1)

        return K.sum(K.dot(arr, weights), axis=1) / scl

    return K.mean(arr, axis=1)





def weighted_loss(y_true, y_pred):

    """

    Will be used as the metric in model.compile()

    ---------------------------------------------

    

    Similar to the custom loss function 'weighted_log_loss()' above

    but with normalized weights, which should be very similar 

    to the official competition metric:

        https://www.kaggle.com/kambarakun/lb-probe-weights-n-of-positives-scoring

    and hence:

        sklearn.metrics.log_loss with sample weights

    """

    

    class_weights = K.variable([2., 1., 1., 1., 1., 1.])

    

    eps = K.epsilon()

    

    y_pred = K.clip(y_pred, eps, 1.0-eps)



    loss = -(        y_true  * K.log(      y_pred)

            + (1.0 - y_true) * K.log(1.0 - y_pred))

    

    loss_samples = _normalized_weighted_average(loss, class_weights)

    

    return K.mean(loss_samples)





def weighted_log_loss_metric(trues, preds):

    """

    Will be used to calculate the log loss 

    of the validation set in PredictionCheckpoint()

    ------------------------------------------

    """

    class_weights = [2., 1., 1., 1., 1., 1.]

    

    epsilon = 1e-7

    

    preds = np.clip(preds, epsilon, 1-epsilon)

    loss = trues * np.log(preds) + (1 - trues) * np.log(1 - preds)

    loss_samples = np.average(loss, axis=1, weights=class_weights)



    return - loss_samples.mean()



# Radam Optimizer

from keras.optimizers import Optimizer



# Ported from https://github.com/LiyuanLucasLiu/RAdam/blob/master/radam.py

class RectifiedAdam(Optimizer):

    """RectifiedAdam optimizer.

    Default parameters follow those provided in the original paper.

    # Arguments

        lr: float >= 0. Learning rate.

        final_lr: float >= 0. Final learning rate.

        beta_1: float, 0 < beta < 1. Generally close to 1.

        beta_2: float, 0 < beta < 1. Generally close to 1.

        gamma: float >= 0. Convergence speed of the bound function.

        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.

        decay: float >= 0. Learning rate decay over each update.

        weight_decay: Weight decay weight.

        amsbound: boolean. Whether to apply the AMSBound variant of this

            algorithm.

    # References

        - [On the Variance of the Adaptive Learning Rate and Beyond]

          (https://arxiv.org/abs/1908.03265)

        - [Adam - A Method for Stochastic Optimization]

          (https://arxiv.org/abs/1412.6980v8)

        - [On the Convergence of Adam and Beyond]

          (https://openreview.net/forum?id=ryQu7f-RZ)

    """



    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,

                 epsilon=None, decay=0., weight_decay=0.0, **kwargs):

        super(RectifiedAdam, self).__init__(**kwargs)



        with K.name_scope(self.__class__.__name__):

            self.iterations = K.variable(0, dtype='int64', name='iterations')

            self.learning_rate = K.variable(lr, name='lr')

            self.beta_1 = K.variable(beta_1, name='beta_1')

            self.beta_2 = K.variable(beta_2, name='beta_2')

            self.decay = K.variable(decay, name='decay')



        if epsilon is None:

            epsilon = K.epsilon()

        self.epsilon = epsilon

        self.initial_decay = decay



        self.weight_decay = float(weight_decay)



    def get_updates(self, loss, params):

        grads = self.get_gradients(loss, params)

        self.updates = [K.update_add(self.iterations, 1)]



        lr = self.learning_rate

        if self.initial_decay > 0:

            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,

                                                      K.dtype(self.decay))))



        t = K.cast(self.iterations, K.floatx()) + 1



        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        self.weights = [self.iterations] + ms + vs



        for p, g, m, v in zip(params, grads, ms, vs):

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g

            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)



            beta2_t = self.beta_2 ** t

            N_sma_max = 2 / (1 - self.beta_2) - 1

            N_sma = N_sma_max - 2 * t * beta2_t / (1 - beta2_t)



            # apply weight decay

            if self.weight_decay != 0.:

                p_wd = p - self.weight_decay * lr * p

            else:

                p_wd = None



            if p_wd is None:

                p_ = p

            else:

                p_ = p_wd



            def gt_path():

                step_size = lr * K.sqrt(

                    (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max /

                    (N_sma_max - 2)) / (1 - self.beta_1 ** t)



                denom = K.sqrt(v_t) + self.epsilon

                p_t = p_ - step_size * (m_t / denom)



                return p_t



            def lt_path():

                step_size = lr / (1 - self.beta_1 ** t)

                p_t = p_ - step_size * m_t



                return p_t



            p_t = K.switch(N_sma > 5, gt_path, lt_path)



            self.updates.append(K.update(m, m_t))

            self.updates.append(K.update(v, v_t))

            new_p = p_t



            # Apply constraints.

            if getattr(p, 'constraint', None) is not None:

                new_p = p.constraint(new_p)



            self.updates.append(K.update(p, new_p))

        return self.updates



    def get_config(self):

        config = {'lr': float(K.get_value(self.learning_rate)),

                  'beta_1': float(K.get_value(self.beta_1)),

                  'beta_2': float(K.get_value(self.beta_2)),

                  'decay': float(K.get_value(self.decay)),

                  'epsilon': self.epsilon,

                  'weight_decay': self.weight_decay}

        base_config = super(RectifiedAdam, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


class PredictionCheckpoint(keras.callbacks.Callback):

    

    def __init__(self, test_df, valid_df, 

                 test_images_dir=test_images_dir, 

                 valid_images_dir=train_images_dir, 

                 batch_size=32, input_size=(224, 224, 3)):

        

        self.test_df = test_df

        self.valid_df = valid_df

        self.test_images_dir = test_images_dir

        self.valid_images_dir = valid_images_dir

        self.batch_size = batch_size

        self.input_size = input_size

        

    def on_train_begin(self, logs={}):

        self.test_predictions = []

        self.valid_predictions = []

        

    def on_epoch_end(self,batch, logs={}):

        self.test_predictions.append(

            self.model.predict_generator(

                DataGenerator(self.test_df.index, None, self.batch_size, self.input_size, self.test_images_dir), verbose=2)[:len(self.test_df)])

        

        # Commented out to save time

        self.valid_predictions.append(

             self.model.predict_generator(

                 DataGenerator(self.valid_df.index, None, self.batch_size, self.input_size, self.valid_images_dir), verbose=2)[:len(self.valid_df)])

        

        print("validation weighted loss: %.4f" %

               weighted_log_loss_metric(self.valid_df.values, 

                                    np.average(self.valid_predictions, axis=0, 

                                               weights=[2**i for i in range(len(self.valid_predictions))])))

        

        # here you could also save the predictions with np.save()





class MyDeepModel:

    

    def __init__(self, engine, input_dims, batch_size=5, num_epochs=4, learning_rate=1e-3, 

                 decay_rate=1.0, decay_steps=1, weights="imagenet", verbose=1):

        

        self.engine = engine

        self.input_dims = input_dims

        self.batch_size = batch_size

        self.num_epochs = num_epochs

        self.learning_rate = learning_rate

        self.decay_rate = decay_rate

        self.decay_steps = decay_steps

        self.weights = weights

        self.verbose = verbose

        self._build()



    def _build(self):

        

        

        engine = self.engine(include_top=False, weights=self.weights, input_shape=self.input_dims,

                             backend = keras.backend, layers = keras.layers,

                             models = keras.models, utils = keras.utils)

        

        x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(engine.output)

#         x = keras.layers.Dropout(0.2)(x)

#         x = keras.layers.Dense(keras.backend.int_shape(x)[1], activation="relu", name="dense_hidden_1")(x)

#         x = keras.layers.Dropout(0.1)(x)

        out = keras.layers.Dense(6, activation="sigmoid", name='dense_output')(x)



        self.model = keras.models.Model(inputs=engine.input, outputs=out)



        self.model.compile(loss="binary_crossentropy", optimizer=RectifiedAdam(), metrics=[weighted_loss])

    



    def fit_and_predict(self, train_df, valid_df, test_df):

        

        # callbacks

        pred_history = PredictionCheckpoint(test_df, valid_df, input_size=self.input_dims)

        #checkpointer = keras.callbacks.ModelCheckpoint(filepath='%s-{epoch:02d}.hdf5' % self.engine.__name__, verbose=1, save_weights_only=True, save_best_only=False)

        scheduler = keras.callbacks.LearningRateScheduler(lambda epoch: self.learning_rate * pow(self.decay_rate, floor(epoch / self.decay_steps)))

        

        self.model.fit_generator(

            DataGenerator(

                train_df.index, 

                train_df, 

                self.batch_size, 

                self.input_dims, 

                train_images_dir

            ),

            epochs=self.num_epochs,

            verbose=self.verbose,

            use_multiprocessing=True,

            workers=4,

            callbacks=[pred_history, scheduler]

        )

        

        return pred_history

    

    def save(self, path):

        self.model.save_weights(path)

    

    def load(self, path):

        self.model.load_weights(path)
def read_testset(filename="../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_sample_submission.csv"):

    df = pd.read_csv(filename)

    df["Image"] = df["ID"].str.slice(stop=12)

    df["Diagnosis"] = df["ID"].str.slice(start=13)

    

    df = df.loc[:, ["Label", "Diagnosis", "Image"]]

    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)

    

    return df



def read_trainset(filename="../input/rsna-intracranial-hemorrhage-detection/rsna-intracranial-hemorrhage-detection/stage_2_train.csv"):

    df = pd.read_csv(filename)

    df["Image"] = df["ID"].str.slice(stop=12)

    df["Diagnosis"] = df["ID"].str.slice(start=13)

    duplicates_to_remove = [56346, 56347, 56348, 56349,

                            56350, 56351, 1171830, 1171831,

                            1171832, 1171833, 1171834, 1171835,

                            3705312, 3705313, 3705314, 3705315,

                            3705316, 3705317, 3842478, 3842479,

                            3842480, 3842481, 3842482, 3842483 ]

    df = df.drop(index = duplicates_to_remove)

    df = df.reset_index(drop = True)    

    df = df.loc[:, ["Label", "Diagnosis", "Image"]]

    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)

    return df



    

test_df = read_testset()

df = read_trainset()
df.head(3)
test_df.head(3)
# train set (00%) and validation set (10%)

ss = ShuffleSplit(n_splits=10, test_size=0.1, random_state=42).split(df.index)



# lets go for the first fold only

train_idx, valid_idx = next(ss)



# obtain model

model = MyDeepModel(engine=InceptionV3, input_dims=(256, 256, 3), batch_size=32, learning_rate=5e-4,

                    num_epochs=2, decay_rate=0.8, decay_steps=1, weights="imagenet", verbose=1)



model.load('../input/rsna-models/model2.h5')



# obtain test + validation predictions (history.test_predictions, history.valid_predictions)

history = model.fit_and_predict(df.iloc[train_idx], df.iloc[valid_idx], test_df)



model.save('model2_stage2.h5')
test_df.iloc[:, :] = np.average(history.test_predictions, axis=0, weights=[1,2])



test_df = test_df.stack().reset_index()



test_df.insert(loc=0, column='ID', value=test_df['Image'].astype(str) + "_" + test_df['Diagnosis'])



test_df = test_df.drop(["Image", "Diagnosis"], axis=1)



test_df.to_csv('submission.csv', index=False)