import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

def load_images_and_masks(folder_path, img_size=(256, 256)):
    images, masks = []
    for filename in os.listdir(folder_path):
        if "_mask" not in filename:
            image_path = os.path.join(folder_path, filename)
            mask_path = os.path.join(folder_path, filename[:-4] + "_mask.png")
            if os.path.exists(mask_path):
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, img_size) / 255.0
                mask = cv2.resize(mask, img_size) / 255.0
                images.append(image)
                masks.append(mask)
    return np.expand_dims(np.array(images), -1), np.expand_dims(np.array(masks), -1)

def unet_model(input_size=(256, 256, 1)):
    inputs = Input(input_size)
    c1 = Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D()(c1)
    c2 = Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D()(c2)
    c3 = Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(64, 3, activation='relu', padding='same')(c3)
    p3 = MaxPooling2D()(c3)
    c4 = Conv2D(128, 3, activation='relu', padding='same')(p3)
    c4 = Conv2D(128, 3, activation='relu', padding='same')(c4)
    u5 = UpSampling2D()(c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(64, 3, activation='relu', padding='same')(u5)
    c5 = Conv2D(64, 3, activation='relu', padding='same')(c5)
    u6 = UpSampling2D()(c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(32, 3, activation='relu', padding='same')(u6)
    c6 = Conv2D(32, 3, activation='relu', padding='same')(c6)
    u7 = UpSampling2D()(c6)
    u7 = concatenate([u7, c1])
    c7 = Conv2D(16, 3, activation='relu', padding='same')(u7)
    c7 = Conv2D(16, 3, activation='relu', padding='same')(c7)
    outputs = Conv2D(1, 1, activation='sigmoid')(c7)
    return Model(inputs, outputs)

if __name__ == "__main__":
    path = "PATH_TO_YOUR_DATASET"  # Replace with actual path
    X, Y = load_images_and_masks(path)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = unet_model()
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=8)
    model.save("unet_model.h5")