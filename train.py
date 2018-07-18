#Imports

import tflearn
from tflearn.data_utils import shuffle
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import pickle

#Loading dataset
X_train, Y_train, X_test, Y_test = pickle.load(open("full_dataset.pkl", "rb"))

#Shuffle dataset
X_train, Y_train = shuffle(X_train, Y_train)

#Normalise dataset
preprocessed_img = ImagePreprocessing()
preprocessed_img.add_featurewise_zero_center()
preprocessed_img.add_featurewise_stdnorm()

#Augmenting dataset
augmented_image = ImageAugmentation()
augmented_image.add_random_flip_leftright()
augmented_image.add_random_rotation(max_angle=25.)
augmented_image.add_random_blur(sigma_max=3.)

#Network

neural_net = input_data(shape=[None, 32, 32, 3], data_preprocessing=preprocessed_img,
                        data_augmentation=augmented_image)

neural_net = conv_2d(neural_net, 32, 3, activation='relu')

neural_net = max_pool_2d(neural_net, 2)

neural_net = conv_2d(neural_net, 64, 3, activation='relu')

neural_net = conv_2d(neural_net, 64, 3, activation='relu')

neural_net = max_pool_2d(neural_net, 2)

neural_net = fully_connected(neural_net, 512, activation='relu')

neural_net = dropout(neural_net, 0.5)

neural_net = fully_connected(neural_net, 2, activation='softmax')

## method of training our neural_net
neural_net = regression(neural_net, optimizer = 'adam',
                        loss='categorical_crossentropy',
                        learning_rate=0.001)

## create model object
model = tflearn.DNN(neural_net, tensorboard_verbose=0, checkpoint_path='storeCheckPoints.tfl.ckpt')

##start training
model.fit(X_train, Y_train, n_epoch=5, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=96,
          snapshot_epoch=True, run_id='bird-classifier')

#Save model to a file
model.save("bird-classifier-model.tfl")

print("Training complete and Model saved..!")














