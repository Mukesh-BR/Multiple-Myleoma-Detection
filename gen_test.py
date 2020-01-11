import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import warnings
import os
import keras
import cv2
import matplotlib.pyplot as plt
from loss import dice_coef_loss,dice_coef
from preprocess import preprocess_mask,preprocess_image
import numpy as np
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def unet(input_size):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    # conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    # conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    # drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(pool4))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(1, 1, activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    #conv10 = Conv2D(3, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv9)

    
    #model.summary()


    return model

model=unet((256,256,3))
model.summary()

# training_data_x=[]
# training_data_y=[]
# test_data_x=[]
# test_data_y=[]
# og_path="//content//drive//My Drive//abc//pqr"
# test_path="//content//drive//My Drive//abc//pqr"
# CATEGORIES=["og","mask","test_og","test_mask"]
# def create_dataset():
    
#      path=os.path.join(og_path,CATEGORIES[0])
#      for img in os.listdir(path):
#          img_array=cv2.imread(os.path.join(path,img))
#          training_data_x.append(img_array)
#      path=os.path.join(og_path,CATEGORIES[1])
#      for img in os.listdir(path):
#          img_array=cv2.imread(os.path.join(path,img))
#          training_data_y.append(img_array)
#      path=os.path.join(og_path,CATEGORIES[2])
#      for img in os.listdir(path):
#          img_array=cv2.imread(os.path.join(path,img))
#          test_data_x.append(img_array)
#      path=os.path.join(og_path,CATEGORIES[3])
#      for img in os.listdir(path):
#          img_array=cv2.imread(os.path.join(path,img))
#          test_data_y.append(img_array)
         

        
# create_dataset()
# training_data_x=np.asarray(training_data_x)
# training_data_y=np.asarray(training_data_y)
# test_data_x=np.asarray(test_data_x)
# test_data_y=np.asarray(test_data_y)


data_gen_args_mask = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2,preprocessing_function=preprocess_mask)

data_gen_args_image = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2,
                     preprocessing_function=preprocess_image)

image_datagen = keras.preprocessing.image.ImageDataGenerator(**data_gen_args_image)
mask_datagen = keras.preprocessing.image.ImageDataGenerator(**data_gen_args_mask)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1

image_generator = image_datagen.flow_from_directory(
    '/home/team6/Project/MiMM_SBILab/patches/train/images',
    class_mode=None,
    target_size=(256,256),
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(
    '/home/team6/Project/MiMM_SBILab/patches/train/masks',
    class_mode=None,
    color_mode="grayscale",
    target_size=(256, 256),
    seed=seed)

#combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)
print(len(image_generator))
model.compile(optimizer="adam",loss=dice_coef_loss,metrics=["accuracy",dice_coef])

# callbacks = [
#     keras.callbacks.EarlyStopping(monitor='loss', patience=25, verbose=1),
#     keras.callbacks.ModelCheckpoint("Resnet_50_{epoch:03d}.hdf5", monitor='loss', verbose=1, mode='auto'),
#     keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, verbose=1, mode='auto', epsilon=0.01, cooldown=0, min_lr=1e-6),
#     keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
#     #NotifyCB
# ]
model.load_weights("Unets_50_019.hdf5")
print("Loaded weights")
# model.fit_generator(
#     train_generator,
#     steps_per_epoch=1700,
#     epochs=100,
#     initial_epoch=50,
#     callbacks=callbacks)

ans=model.predict_generator(image_generator,steps=10,verbose=1)
print(ans.shape)
count=0
np.save('preds',ans)

for i in range(ans.shape[0]):
    for j in range(ans.shape[1]):
        for k in range(ans.shape[2]):
            for l in range(ans.shape[3]):
                if(ans[i][j][k][l]!=0):
                    count+=1
print(count)
'''
# cv2.imwrite("Sample.jpg",ans)
# cv2.imwrite("preprocess.jpg",img_result)
# h1.fit(training_data_x,training_data_y,epochs=10,batch_size=3)
# pred=h1.evaluate(test_data_x,test_data_y)
# print("loss"+str(pred[0]))
# print("acc"+str(pred[1]))
