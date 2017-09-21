import csv
import numpy as np
import matplotlib.image as mpimg
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation, MaxPooling2D, Cropping2D, Dropout
import matplotlib.pyplot as plt
from keras.layers import Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import gc

lines=[]
angles=[]
images=[]

with open('data/driving_log.csv')as f:
    reader=csv.reader(f)
    next(reader)
    for line in reader:
        lines.append(line)

# samples freqwency analysis
def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

def plot_angle_distribution(lines):
        data = []
        for i in range(len(lines)):
            data.append(float(lines[i][3])*25.)
        data=np.asarray(data)
        binwidth=1
        n, bins, patches = plt.hist(data,bins=np.arange(data.min(), data.max()+ binwidth,binwidth)-0.5* binwidth)
        plt.xlabel('steering angles (degree)')
        plt.ylabel('occurence')
        plt.title('Histogram of steering')
        plt.show()
plot_angle_distribution(lines)

# processing data:
print('processing data')
path='data/IMG/'
correction=-0.3
for line in lines:
    for i in range(3):
        source_path=line[i]
        file_name=source_path.split('/')[-1]
        image_name=path+file_name
        image=mpimg.imread(image_name)
        images.append(image)                # adding image to images list
        images.append(np.fliplr(image))     # adding flipped images
        angle=float(line[3])+(i*3./2*(i-5./3))* correction # in bracket function returning 0,-1,1 for 0,1,2 (center , left, right)
        angles.append(angle)                # adding measurement to measurements list
        angles.append(-angle)               # adding flipped measurements
x=np.array(images)
y=np.array(angles)

# model definition :
model= Sequential()
model.add(Cropping2D(cropping=((70,81), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Convolution2D(1,9,9, border_mode='same', subsample=(2, 2),activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(1))
model.summary()
model.compile(loss='mse',optimizer='adam')

# model launch
history = model.fit(x, y, epochs=10, verbose=1,  validation_split=0.2,  shuffle=True)

# display progress
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model_simplified.h5')
gc.collect()
