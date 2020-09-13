from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import numpy as np


class Encoder:
    def __init__(self):
        input_img = Input(shape=(320, 176, 1))

        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(2, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(1, (3, 3), activation='relu', padding='same')(x)
        encoded = MaxPooling2D((4, 1), padding='same')(x)

        x = Conv2D(1, (3, 3), activation='relu', padding='same')(encoded)
        x = UpSampling2D((4, 1))(x)
        x = Conv2D(2, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(4, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

        self.autoencoder = Model(input_img, decoded)
        self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        self.autoencoder.load_weights("Weights/AutoencoderBDD100kCombination.h5")
        self.encoder = Model(input_img, encoded)

    def Encode_img(self, img):
        img = np.expand_dims(np.expand_dims(img/255,axis = 0),axis = 3)
        encoded_imgs = self.encoder.predict(img)
        return np.squeeze(encoded_imgs.flatten())

    def Encode_img_not_flatted(self, img):
        img = np.expand_dims(np.expand_dims(img/255,axis = 0),axis = 3)
        encoded_imgs = self.encoder.predict(img)
        return np.squeeze(encoded_imgs)

    def Autoencode(self, img):
        img = np.expand_dims(np.expand_dims(img/255,axis = 0),axis = 3)
        encoded_imgs = self.autoencoder.predict(img)
        return np.squeeze(encoded_imgs)
