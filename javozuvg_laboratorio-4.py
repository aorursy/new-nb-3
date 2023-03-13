import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn import svm, datasets

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.utils.multiclass import unique_labels



import os

print(os.listdir("../input"))
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential, load_model

from keras.layers import (Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D,

                          BatchNormalization, Input, Conv2D, GlobalAveragePooling2D,concatenate,Concatenate)

from keras.callbacks import ModelCheckpoint

from keras import metrics

from keras.optimizers import Adam 

from keras import backend as K

import keras

from keras.models import Model, Sequential
import matplotlib.pyplot as plt

import skimage.io

from skimage.transform import resize

from imgaug import augmenters as iaa

from tqdm import tqdm

import PIL

from PIL import Image, ImageOps

import cv2

from sklearn.utils import class_weight, shuffle

from keras.losses import binary_crossentropy, categorical_crossentropy

#from keras.applications.resnet50 import preprocess_input

from keras.applications.densenet import DenseNet121,DenseNet169

import keras.backend as K

import tensorflow as tf

from sklearn.metrics import f1_score, fbeta_score, cohen_kappa_score

from keras.utils import Sequence

from keras.utils import to_categorical

from sklearn.model_selection import train_test_split

import imgaug as ia



WORKERS = 2

CHANNEL = 3



import warnings

warnings.filterwarnings("ignore")

SIZE = 300

NUM_CLASSES = 5
df_train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

df_test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')
x = df_train['id_code']

y = df_train['diagnosis']



x, y = shuffle(x, y, random_state=8)

y.hist()
y = to_categorical(y, num_classes=NUM_CLASSES)

train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.15,

                                                      stratify=y, random_state=8)

print(train_x.shape)

print(train_y.shape)

print(valid_x.shape)

print(valid_y.shape)
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

seq = iaa.Sequential(

        [

            # apply the following augmenters to most images

            iaa.Fliplr(0.5), # horizontally flip 50% of all images

            iaa.Flipud(0.2), # vertically flip 20% of all images

            sometimes(iaa.Affine(

                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to 80-120% of their size, individually per axis

                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)

                rotate=(-10, 10), # rotate by -45 to +45 degrees

                shear=(-5, 5), # shear by -16 to +16 degrees

                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)

                cval=(0, 255), # if mode is constant, use a cval between 0 and 255

                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)

            )),

            # execute 0 to 5 of the following (less important) augmenters per image

            # don't execute all of them, as that would often be way too strong

            iaa.SomeOf((0, 5),

                [

                    sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation

                    iaa.OneOf([

                        iaa.GaussianBlur((0, 1.0)), # blur images with a sigma between 0 and 3.0

                        iaa.AverageBlur(k=(3, 5)), # blur image using local means with kernel sizes between 2 and 7

                        iaa.MedianBlur(k=(3, 5)), # blur image using local medians with kernel sizes between 2 and 7

                    ]),

                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1)), # sharpen images

                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images

                    # search either for all edges or for directed edges,

                    # blend the result with the original image using a blobby mask

                    iaa.SimplexNoiseAlpha(iaa.OneOf([

                        iaa.EdgeDetect(alpha=(0.5, 1.0)),

                        iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),

                    ])),

                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255), per_channel=0.5), # add gaussian noise to images

                    iaa.OneOf([

                        iaa.Dropout((0.01, 0.05), per_channel=0.5), # randomly remove up to 10% of the pixels

                        iaa.CoarseDropout((0.01, 0.03), size_percent=(0.01, 0.02), per_channel=0.2),

                    ]),

                    iaa.Invert(0.01, per_channel=True), # invert color channels

                    iaa.Add((-2, 2), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)

                    iaa.AddToHueAndSaturation((-1, 1)), # change hue and saturation

                    # either change the brightness of the whole image (sometimes

                    # per channel) or change the brightness of subareas

                    iaa.OneOf([

                        iaa.Multiply((0.9, 1.1), per_channel=0.5),

                        iaa.FrequencyNoiseAlpha(

                            exponent=(-1, 0),

                            first=iaa.Multiply((0.9, 1.1), per_channel=True),

                            second=iaa.ContrastNormalization((0.9, 1.1))

                        )

                    ]),

                    sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)

                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around

                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))

                ],

                random_order=True

            )

        ],

        random_order=True)
class My_Generator(Sequence):



    def __init__(self, image_filenames, labels,

                 batch_size, is_train=True,

                 mix=False, augment=False):

        self.image_filenames, self.labels = image_filenames, labels

        self.batch_size = batch_size

        self.is_train = is_train

        self.is_augment = augment

        if(self.is_train):

            self.on_epoch_end()

        self.is_mix = mix



    def __len__(self):

        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))



    def __getitem__(self, idx):

        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]



        if(self.is_train):

            return self.train_generate(batch_x, batch_y)

        return self.valid_generate(batch_x, batch_y)



    def on_epoch_end(self):

        if(self.is_train):

            self.image_filenames, self.labels = shuffle(self.image_filenames, self.labels)

        else:

            pass

    

    def mix_up(self, x, y):

        lam = np.random.beta(0.2, 0.4)

        ori_index = np.arange(int(len(x)))

        index_array = np.arange(int(len(x)))

        np.random.shuffle(index_array)        

        

        mixed_x = lam * x[ori_index] + (1 - lam) * x[index_array]

        mixed_y = lam * y[ori_index] + (1 - lam) * y[index_array]

        

        return mixed_x, mixed_y



    def train_generate(self, batch_x, batch_y):

        batch_images = []

        for (sample, label) in zip(batch_x, batch_y):

            img = cv2.imread('../input/aptos2019-blindness-detection/train_images/'+sample+'.png')

            img = cv2.resize(img, (SIZE, SIZE))

            if(self.is_augment):

                img = seq.augment_image(img)

            batch_images.append(img)

        batch_images = np.array(batch_images, np.float32) / 255

        batch_y = np.array(batch_y, np.float32)

        if(self.is_mix):

            batch_images, batch_y = self.mix_up(batch_images, batch_y)

        return batch_images, batch_y



    def valid_generate(self, batch_x, batch_y):

        batch_images = []

        for (sample, label) in zip(batch_x, batch_y):

            img = cv2.imread('../input/aptos2019-blindness-detection/train_images/'+sample+'.png')

            img = cv2.resize(img, (SIZE, SIZE))

            batch_images.append(img)

        batch_images = np.array(batch_images, np.float32) / 255

        batch_y = np.array(batch_y, np.float32)

        return batch_images, batch_y
def create_simple_model(input_shape, n_out):

    model = keras.Sequential([

        keras.layers.Flatten(input_shape=input_shape),

        keras.layers.Dense(128, activation='relu'),

        keras.layers.Dense(5, activation='relu')

    ])

    return model
# create callbacks list

from keras.callbacks import (ModelCheckpoint, LearningRateScheduler,

                             EarlyStopping, ReduceLROnPlateau,CSVLogger)



epochs = 30; batch_size = 32

checkpoint = ModelCheckpoint('../working/densenet_.h5', monitor='val_loss', verbose=1, 

                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, 

                                   verbose=1, mode='auto', epsilon=0.0001)

early = EarlyStopping(monitor="val_loss", 

                      mode="min", 

                      patience=9)



csv_logger = CSVLogger(filename='../working/training_log.csv',

                       separator=',',

                       append=True)



train_generator = My_Generator(train_x, train_y, 128, is_train=True)

train_mixup = My_Generator(train_x, train_y, batch_size, is_train=True, mix=False, augment=True)

valid_generator = My_Generator(valid_x, valid_y, batch_size, is_train=False)



s_model = create_simple_model(

    input_shape=(SIZE,SIZE,3), 

    n_out=5)
# For a multi-class classification problem

s_model.compile(optimizer='rmsprop',

              loss='categorical_crossentropy',

              metrics=['accuracy'])
def kappa_loss(y_true, y_pred, y_pow=2, eps=1e-12, N=5, bsize=32, name='kappa'):

    """A continuous differentiable approximation of discrete kappa loss.

        Args:

            y_pred: 2D tensor or array, [batch_size, num_classes]

            y_true: 2D tensor or array,[batch_size, num_classes]

            y_pow: int,  e.g. y_pow=2

            N: typically num_classes of the model

            bsize: batch_size of the training or validation ops

            eps: a float, prevents divide by zero

            name: Optional scope/name for op_scope.

        Returns:

            A tensor with the kappa loss."""



    with tf.name_scope(name):

        y_true = tf.to_float(y_true)

        repeat_op = tf.to_float(tf.tile(tf.reshape(tf.range(0, N), [N, 1]), [1, N]))

        repeat_op_sq = tf.square((repeat_op - tf.transpose(repeat_op)))

        weights = repeat_op_sq / tf.to_float((N - 1) ** 2)

    

        pred_ = y_pred ** y_pow

        try:

            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [-1, 1]))

        except Exception:

            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [bsize, 1]))

    

        hist_rater_a = tf.reduce_sum(pred_norm, 0)

        hist_rater_b = tf.reduce_sum(y_true, 0)

    

        conf_mat = tf.matmul(tf.transpose(pred_norm), y_true)

    

        nom = tf.reduce_sum(weights * conf_mat)

        denom = tf.reduce_sum(weights * tf.matmul(

            tf.reshape(hist_rater_a, [N, 1]), tf.reshape(hist_rater_b, [1, N])) /

                              tf.to_float(bsize))

    

        return nom*0.5 / (denom + eps) + categorical_crossentropy(y_true, y_pred)*0.5
from keras.callbacks import Callback

class QWKEvaluation(Callback):

    def __init__(self, validation_data=(), batch_size=64, interval=1):

        super(Callback, self).__init__()



        self.interval = interval

        self.batch_size = batch_size

        self.valid_generator, self.y_val = validation_data

        self.history = []



    def on_epoch_end(self, epoch, logs={}):

        if epoch % self.interval == 0:

            y_pred = self.model.predict_generator(generator=self.valid_generator,

                                                  steps=np.ceil(float(len(self.y_val)) / float(self.batch_size)),

                                                  workers=1, use_multiprocessing=False,

                                                  verbose=1)

            def flatten(y):

                return np.argmax(y, axis=1).reshape(-1)

            

            score = cohen_kappa_score(flatten(self.y_val),

                                      flatten(y_pred),

                                      labels=[0,1,2,3,4],

                                      weights='quadratic')

            print("\n epoch: %d - QWK_score: %.6f \n" % (epoch+1, score))

            self.history.append(score)

            if score >= max(self.history):

                print('saving checkpoint: ', score)

                self.model.save('../working/s_densenet_bestqwk.h5')



qwk = QWKEvaluation(validation_data=(valid_generator, valid_y),

                    batch_size=batch_size, interval=1)
for layer in s_model.layers:

    layer.trainable = False



for i in range(-3,0):

    s_model.layers[i].trainable = True



s_model.compile(

    loss='categorical_crossentropy',

    optimizer=Adam(1e-3))
s_model.fit_generator(

    train_generator,

    steps_per_epoch=np.ceil(float(len(train_y)) / float(128)),

    epochs=1,

    workers=WORKERS, use_multiprocessing=True,

    verbose=1,

    callbacks=[qwk])
submit = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

s_model.load_weights('../working/s_densenet_bestqwk.h5')
s_predicted = []

for i, (g, name) in tqdm(enumerate(enumerate(valid_x))):

    path = os.path.join('../input/aptos2019-blindness-detection/train_images/', name+'.png')

    image = cv2.imread(path)

    image = cv2.resize(image, (SIZE, SIZE))

    X = np.array((image[np.newaxis])/255)

    score_predict=((s_model.predict(X).ravel()*s_model.predict(X[:, ::-1, :, :]).ravel()*s_model.predict(X[:, ::-1, ::-1, :]).ravel()*s_model.predict(X[:, :, ::-1, :]).ravel())**0.25).tolist()

    label_predict = np.argmax(score_predict)

    s_predicted.append(str(label_predict))
def create_model(input_shape, n_out):

    input_tensor = Input(shape=input_shape)

    base_model = DenseNet121(include_top=False,

                   weights=None,

                   input_tensor=input_tensor)

    x = GlobalAveragePooling2D()(base_model.output)

    x = Dropout(0.5)(x)

    x = Dense(1024, activation='relu')(x)

    x = Dropout(0.5)(x)

    final_output = Dense(n_out, activation='softmax', name='final_output')(x)

    model = Model(input_tensor, final_output) 

    return model
# create callbacks list

from keras.callbacks import (ModelCheckpoint, LearningRateScheduler,

                             EarlyStopping, ReduceLROnPlateau,CSVLogger)



epochs = 30; batch_size = 32

checkpoint = ModelCheckpoint('../working/densenet_.h5', monitor='val_loss', verbose=1, 

                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, 

                                   verbose=1, mode='auto', epsilon=0.0001)

early = EarlyStopping(monitor="val_loss", 

                      mode="min", 

                      patience=9)



csv_logger = CSVLogger(filename='../working/training_log.csv',

                       separator=',',

                       append=True)



train_generator = My_Generator(train_x, train_y, 128, is_train=True)

train_mixup = My_Generator(train_x, train_y, batch_size, is_train=True, mix=False, augment=True)

valid_generator = My_Generator(valid_x, valid_y, batch_size, is_train=False)



model = create_model(

    input_shape=(SIZE,SIZE,3), 

    n_out=5)
def kappa_loss(y_true, y_pred, y_pow=2, eps=1e-12, N=5, bsize=32, name='kappa'):

    """A continuous differentiable approximation of discrete kappa loss.

        Args:

            y_pred: 2D tensor or array, [batch_size, num_classes]

            y_true: 2D tensor or array,[batch_size, num_classes]

            y_pow: int,  e.g. y_pow=2

            N: typically num_classes of the model

            bsize: batch_size of the training or validation ops

            eps: a float, prevents divide by zero

            name: Optional scope/name for op_scope.

        Returns:

            A tensor with the kappa loss."""



    with tf.name_scope(name):

        y_true = tf.to_float(y_true)

        repeat_op = tf.to_float(tf.tile(tf.reshape(tf.range(0, N), [N, 1]), [1, N]))

        repeat_op_sq = tf.square((repeat_op - tf.transpose(repeat_op)))

        weights = repeat_op_sq / tf.to_float((N - 1) ** 2)

    

        pred_ = y_pred ** y_pow

        try:

            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [-1, 1]))

        except Exception:

            pred_norm = pred_ / (eps + tf.reshape(tf.reduce_sum(pred_, 1), [bsize, 1]))

    

        hist_rater_a = tf.reduce_sum(pred_norm, 0)

        hist_rater_b = tf.reduce_sum(y_true, 0)

    

        conf_mat = tf.matmul(tf.transpose(pred_norm), y_true)

    

        nom = tf.reduce_sum(weights * conf_mat)

        denom = tf.reduce_sum(weights * tf.matmul(

            tf.reshape(hist_rater_a, [N, 1]), tf.reshape(hist_rater_b, [1, N])) /

                              tf.to_float(bsize))

    

        return nom*0.5 / (denom + eps) + categorical_crossentropy(y_true, y_pred)*0.5
from keras.callbacks import Callback

class QWKEvaluation(Callback):

    def __init__(self, validation_data=(), batch_size=64, interval=1):

        super(Callback, self).__init__()



        self.interval = interval

        self.batch_size = batch_size

        self.valid_generator, self.y_val = validation_data

        self.history = []



    def on_epoch_end(self, epoch, logs={}):

        if epoch % self.interval == 0:

            y_pred = self.model.predict_generator(generator=self.valid_generator,

                                                  steps=np.ceil(float(len(self.y_val)) / float(self.batch_size)),

                                                  workers=1, use_multiprocessing=False,

                                                  verbose=1)

            def flatten(y):

                return np.argmax(y, axis=1).reshape(-1)

            

            score = cohen_kappa_score(flatten(self.y_val),

                                      flatten(y_pred),

                                      labels=[0,1,2,3,4],

                                      weights='quadratic')

            print("\n epoch: %d - QWK_score: %.6f \n" % (epoch+1, score))

            self.history.append(score)

            if score >= max(self.history):

                print('saving checkpoint: ', score)

                self.model.save('../working/densenet_bestqwk.h5')



qwk = QWKEvaluation(validation_data=(valid_generator, valid_y),

                    batch_size=batch_size, interval=1)
for layer in model.layers:

    layer.trainable = False



for i in range(-3,0):

    model.layers[i].trainable = True



model.compile(

    loss='categorical_crossentropy',

    optimizer=Adam(1e-3))
model.fit_generator(

    train_generator,

    steps_per_epoch=np.ceil(float(len(train_y)) / float(128)),

    epochs=3,

    workers=WORKERS, use_multiprocessing=True,

    verbose=1,

    callbacks=[qwk])
submit = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

model.load_weights('../working/densenet_bestqwk.h5')

predicted = []

real = []
# reference:https://www.kaggle.com/CVxTz/cnn-starter-nasnet-mobile-0-9709-lb 

for i, name in tqdm(enumerate(submit['id_code'])):

    path = os.path.join('../input/aptos2019-blindness-detection/test_images/', name+'.png')

    image = cv2.imread(path)

    image = cv2.resize(image, (SIZE, SIZE))

    X = np.array((image[np.newaxis])/255)

    score_predict=((model.predict(X).ravel()*model.predict(X[:, ::-1, :, :]).ravel()*model.predict(X[:, ::-1, ::-1, :]).ravel()*model.predict(X[:, :, ::-1, :]).ravel())**0.25).tolist()

    label_predict = np.argmax(score_predict)

    predicted.append(str(label_predict))
submit['predict'] = predicted

submit.head()
predicted = []

for i, (g, name) in tqdm(enumerate(enumerate(valid_x))):

    path = os.path.join('../input/aptos2019-blindness-detection/train_images/', name+'.png')

    image = cv2.imread(path)

    image = cv2.resize(image, (SIZE, SIZE))

    X = np.array((image[np.newaxis])/255)

    score_predict=((model.predict(X).ravel()*model.predict(X[:, ::-1, :, :]).ravel()*model.predict(X[:, ::-1, ::-1, :]).ravel()*model.predict(X[:, :, ::-1, :]).ravel())**0.25).tolist()

    label_predict = np.argmax(score_predict)

    predicted.append(str(label_predict))
predicted = np.asarray(predicted)

s_predicted = np.asarray(s_predicted)

real = np.array(list(map(lambda item: str(list(item).index(1)), valid_y)))
def plot_confusion_matrix(cm,

                          target_names,

                          title='Confusion matrix',

                          cmap=None,

                          normalize=True):

    """

    given a sklearn confusion matrix (cm), make a nice plot



    Arguments

    ---------

    cm:           confusion matrix from sklearn.metrics.confusion_matrix



    target_names: given classification classes such as [0, 1, 2]

                  the class names, for example: ['high', 'medium', 'low']



    title:        the text to display at the top of the matrix



    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm

                  see http://matplotlib.org/examples/color/colormaps_reference.html

                  plt.get_cmap('jet') or plt.cm.Blues



    normalize:    If False, plot the raw numbers

                  If True, plot the proportions



    Usage

    -----

    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by

                                                              # sklearn.metrics.confusion_matrix

                          normalize    = True,                # show proportions

                          target_names = y_labels_vals,       # list of names of the classes

                          title        = best_estimator_name) # title of graph



    Citiation

    ---------

    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html



    """

    import matplotlib.pyplot as plt

    import numpy as np

    import itertools



    accuracy = np.trace(cm) / float(np.sum(cm))

    misclass = 1 - accuracy



    if cmap is None:

        cmap = plt.get_cmap('Blues')



    plt.figure(figsize=(8, 6))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()



    if target_names is not None:

        tick_marks = np.arange(len(target_names))

        plt.xticks(tick_marks, target_names, rotation=45)

        plt.yticks(tick_marks, target_names)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]





    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        if normalize:

            plt.text(j, i, "{:0.4f}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")

        else:

            plt.text(j, i, "{:,}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")





    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    plt.show()
s_cm = confusion_matrix(real, s_predicted)

cm = confusion_matrix(real, predicted)
plot_confusion_matrix(cm=s_cm, 

                      normalize    = False,

                      target_names = ['0', '1', '2', '3', '4'],

                      title        = "Confusion Matrix")
plot_confusion_matrix(cm=cm, 

                      normalize    = False,

                      target_names = ['0', '1', '2', '3', '4'],

                      title        = "Confusion Matrix")