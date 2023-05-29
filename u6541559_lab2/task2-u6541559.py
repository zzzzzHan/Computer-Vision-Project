#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import torch
from torchvision import transforms
import cv2
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.utils import to_categorical


# In[2]:


# import the data in array 
train_img = np.array(np.load('kmnist-train-imgs.npz')['arr_0'],dtype='uint8')
train_label = np.array(np.load('kmnist-train-labels.npz')['arr_0'],dtype='uint8')
test_img = np.array(np.load('kmnist-test-imgs.npz')['arr_0'],dtype='uint8')
test_label = np.array(np.load('kmnist-test-labels.npz')['arr_0'],dtype='uint8')
val_img = np.array(np.load('kmnist-val-imgs.npz')['arr_0'],dtype='uint8')
val_label = np.array(np.load('kmnist-val-labels.npz')['arr_0'],dtype='uint8')


# In[3]:


# Normalization 
transform = transforms.Compose([
    transforms.ToTensor(),  # Tensor normalization  (0,255)~(0,1)
    transforms.Normalize(mean = 0.5, std = 0.5)])  #  normalization  (0,1)~(-1,1)


# In[4]:


normal_train_img =  np.array([transform(train_img[i]).tolist()[0] for i in range(0,len(train_img))])
normal_test_img =  np.array([transform(test_img[i]).tolist()[0] for i in range(0,len(test_img))])
normal_val_img =  np.array([transform(val_img[i]).tolist()[0] for i in range(0,len(val_img))])


# In[5]:


normal_train_img.shape
normal_test_img.shape
normal_val_img.shape


# In[6]:


# Random choose  pictures to do the flipping, the index is random 
number_of_flips = 1000 
random_index = np.random.randint(len(train_img), size=number_of_flips) #random index


# In[7]:


for index in random_index:
    # flip left to right, right to left
    normal_train_img[index] =  cv2.flip(normal_train_img[index], 1) 


# In[8]:


pad_train_img = []
for i in range(0,len(normal_train_img)):
    #pad each of the trainning image with size of 4 zeroes
    pad_train_img.append(np.pad(normal_train_img[i], pad_width=4)) 


# In[9]:


np.array(pad_train_img).shape


# In[10]:


cropped_train_img = []
for i in range(0,len(normal_train_img)):
    # crop the 32*32 image to 28*28 randomly 
    initx, inity = np.random.randint(4),np.random.randint(4) #random index of 0,1,2,3
    cropped_train_img.append(pad_train_img[i][initx:initx+28, inity:inity+28]) #insure the cropped size is 28 


# In[11]:


cropped_train_img = np.array(cropped_train_img) # apply the crop to trainning data


# In[12]:


onehot_train_labels = to_categorical(train_label) # change label to one-hot


# In[13]:


train_images = np.expand_dims(cropped_train_img, axis=3) #reshape the trainning data to 3 dimensions 


# In[14]:


model = models.Sequential()
# 5×5 Convolutional Layer with 32 filters, stride 1 and padding 2 and then  ReLU Activation Layer.
# padding of size 2 is equivelant to 'same'  (28+2*2-5+1==28)
model.add(layers.Conv2D(32, (5, 5), strides = 1, padding ='same', activation='relu', input_shape=(28, 28,1)))

#2×2 Max Pooling Layer with a stride of 2.
model.add(layers.MaxPooling2D((2, 2),strides = 2))

# 3×3 Convolutional Layer with 64 filters, stride 1 and padding 1 then  ReLU Activation Layer.
# padding of size 1 is equivelant to 'same'  (14+1*2-3+1==14)
model.add(layers.Conv2D(64, (3, 3), strides = 1, padding ='same', activation='relu'))

#2×2 Max Pooling Layer with a stride of 2.
model.add(layers.MaxPooling2D((2, 2),strides = 2))

#flatten layers for constructure of fully-connected layer later 
model.add(layers.Flatten())

# Fully-connected layer with 1024 output units. ReLU Activation Layer.
model.add(layers.Dense(1024, activation='relu'))

#Fully-connected layer with 10 output units 
model.add(layers.Dense(10,activation="softmax"))


# In[15]:


model.summary()


# In[16]:


adam = tf.keras.optimizers.Adam(lr = 1e-3, beta_1=0.9, beta_2=0.999)
model.compile(
  adam,
  loss='categorical_crossentropy',
  metrics=['accuracy']
)
r = model.fit(
  train_images,
  onehot_train_labels,
   batch_size =128,
  epochs=15,
  validation_data= (np.expand_dims(normal_val_img, axis=3), to_categorical(val_label))
)


# In[17]:


plt.title('train accuracy with epoches')
plt.plot(r.history['accuracy'])
plt.show()

plt.title('validation accuracy with epoches')
plt.plot(r.history['val_accuracy'])
plt.show()

plt.title('train loss with epoches')
plt.plot(r.history['loss'])
plt.show()

plt.title('validation loss with epoches')
plt.plot(val_loss)
plt.show()


# In[18]:


#New model 
model1 = models.Sequential()


# 5×5 Convolutional Layer with 32 filters, stride 1 and padding 2 and then  ReLU Activation Layer.
# padding of size 2 is equivelant to 'same'  (28+2*2-5+1==28)
model1.add(layers.Conv2D(32, (5, 5), strides = 1, padding ='same', activation='relu', input_shape=(28, 28,1)))

#model1.add(layers.BatchNormalization(epsilon=1e-06 ,momentum=0.9, weights=None))


#2×2 Max Pooling Layer with a stride of 2.
model1.add(layers.MaxPooling2D((2, 2),strides = 2))

# 3×3 Convolutional Layer with 64 filters, stride 1 and padding 1 then  ReLU Activation Layer.
# padding of size 1 is equivelant to 'same'  (14+1*2-3+1==14)
model1.add(layers.Conv2D(64, (3, 3),kernel_regularizer=tf.keras.regularizers.l2(0.01), bias_regularizer=tf.keras.regularizers.l2(0.01), strides = 1, padding ='same', activation='relu'))

#2×2 Max Pooling Layer with a stride of 2.
model1.add(layers.MaxPooling2D((2, 2),strides = 2))

#add more convolution layer
model1.add(layers.Conv2D(32, (2, 2)))
           
#add dropout to prevent overfitting 
model1.add(layers.Dropout(0.1))

#flatten layers for constructure of fully-connected layer later 
model1.add(layers.Flatten())

# Fully-connected layer with 1024 output units. ReLU Activation Layer.
model1.add(layers.Dense(1024, activation='relu'))

#add dropout to prevent overfitting 
model1.add(layers.Dropout(0.1))

#Fully-connected layer with 10 output units 
model1.add(layers.Dense(10,activation="softmax"))



# In[19]:


model1.summary()


# In[22]:


# different set of parameters
#optimizers and learning rate 
adam = tf.keras.optimizers.Adam(lr = 1e-3, beta_1=0.9, beta_2=0.999)
adam1 = tf.keras.optimizers.Adam(lr = 1.5e-3,decay = 0.9)
adam2 = tf.keras.optimizers.Adam(lr = 2e-3,decay = 0.8)
sgd1 = tf.keras.optimizers.SGD(lr=0.01, clipnorm=1.)
RMSprop =tf.keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
 


# In[23]:


#batch size = 16,32,62,128,256  
batch_size1 =  16
batch_size2 = 32
batch_size3 = 64
batch_size4 = 128
batch_size5 = 256  


# In[24]:


model1.compile(
  adam,
  loss='categorical_crossentropy',
  metrics=['accuracy']
)

# To prevent overfitting, seperate the trainning, first trainning 10 epoch, then decide whether to continue 
r = model1.fit(
  train_images,
  onehot_train_labels,
   batch_size =batch_size4,
  epochs=10,
  validation_data= (np.expand_dims(normal_val_img, axis=3), to_categorical(val_label))
)


# In[26]:


# Continue to train from the previous result 
r1 = model1.fit(
  train_images,
  onehot_train_labels,
   batch_size =batch_size4,
  epochs=5,
  validation_data= (np.expand_dims(normal_val_img, axis=3), to_categorical(val_label))
)


# In[28]:


plt.title('train accuracy with epoches')
plt.plot(r.history['accuracy']+r1.history['accuracy'])
plt.show()

plt.title('validation accuracy with epoches')
plt.plot(r.history['val_accuracy']+r1.history['val_accuracy'])
plt.show()

plt.title('train loss with epoches')
plt.plot(r.history['loss']+r1.history['loss'])
plt.show()

plt.title('validation loss with epoches')
plt.plot(r.history['val_loss']+r1.history['val_loss'])
plt.show()


# In[32]:


predicted=model1.predict(np.expand_dims(normal_test_img, axis=3)) 
result  =np.argmax(predicted,axis=1)
num = 0 
for i in range(len(test_label)):
    if test_label[i]==result[i]:
        num+=1
print('testing accuracy for modified model ',num/len(test_label))


# In[31]:


predicted=model.predict(np.expand_dims(normal_test_img, axis=3)) 
result  =np.argmax(predicted,axis=1)
num = 0 
for i in range(len(test_label)):
    if test_label[i]==result[i]:
        num+=1
print('testing accuracy for original model',num/len(test_label))







