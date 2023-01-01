import tensorflow as tf
import matplotlib.pyplot as plt
import pickle as pkl
from split import tf_train_valid_test_split


BATCH_SIZE = 32
IMAGE_SIZE = 256
SUMMARY_FILE_NAME = 'potatoes_report.txt'


# dataset import
dataset = tf.keras.utils.image_dataset_from_directory(
    directory  = 'PlantVillage/Potatoes',
    batch_size = 32 ,
    image_size = (IMAGE_SIZE , IMAGE_SIZE) , 
    shuffle    = True
)

train_ds , valid_ds , test_ds = tf_train_valid_test_split(dataset)

# ---- augmentation and re-scaling --- #

# -- rescale / resize layer
rescale_and_resize = tf.keras.Sequential()
rescale_and_resize.add(tf.keras.layers.Resizing(IMAGE_SIZE , IMAGE_SIZE))
rescale_and_resize.add(tf.keras.layers.Rescaling(1.0/255))


# data_agumentation layer 
data_augmentation = tf.keras.Sequential()
data_augmentation.add(tf.keras.layers.RandomContrast(0.2))
data_augmentation.add(tf.keras.layers.RandomRotation(0.2))
data_augmentation.add(tf.keras.layers.RandomZoom(0.2))


augmented_train_ds = train_ds.map(lambda x , y : (data_augmentation(x , training = True) , y))
train_ds = augmented_train_ds.concatenate(augmented_train_ds).shuffle(100)

# ---  cache and prefetch the dataset 
train_ds = train_ds.prefetch(buffer_size = tf.data.AUTOTUNE).cache()
valid_ds = valid_ds.prefetch(buffer_size = tf.data.AUTOTUNE).cache()
test_ds = test_ds.prefetch(buffer_size = tf.data.AUTOTUNE).cache()


#---- model creation ----#

model = tf.keras.Sequential()

model.add(rescale_and_resize)
model.add(tf.keras.layers.Conv2D(filters = 32 , kernel_size = (3,3) ,
                              activation = 'relu' , input_shape = (BATCH_SIZE , IMAGE_SIZE , IMAGE_SIZE , 3)))
model.add(tf.keras.layers.MaxPooling2D((2 , 2)))

model.add(tf.keras.layers.Conv2D(filters = 64 , kernel_size = (3,3) , activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D((2 , 2)))

model.add(tf.keras.layers.Conv2D(filters = 64 , kernel_size = (5,5) , activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D((2 , 2)))

model.add(tf.keras.layers.Conv2D(filters = 32 , kernel_size = (3,3) , activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D((3 , 3)))
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(64 , activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(3 , activation = 'softmax'))

model.build(input_shape = (BATCH_SIZE , IMAGE_SIZE , IMAGE_SIZE , 3))

model.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate = 1e-3 , decay = 1e-6),
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
    metrics = ['accuracy']
)

training = model.fit(
    train_ds ,
    batch_size=BATCH_SIZE,
    epochs = 6,
    validation_data = valid_ds
    )


plt.figure(figsize = (10 , 10))
plt.subplot(1 , 2 , 1)
plt.plot(training.history['loss'] , label = 'train loss')
plt.plot(training.history['val_loss'] , label = 'validation loss')
plt.legend()
plt.xlabel('epochs')


plt.subplot(2 , 2 , 2)
plt.plot(training.history['accuracy'] , label = 'train accuracy')
plt.plot(training.history['val_accuracy'] , label = 'validation accuracy')
plt.legend()
plt.xlabel('epochs')

plt.show()


pkl.dump(model , open('potatoes_model.obj' , 'wb'))


try:
    # we print the result in a file
    with open(SUMMARY_FILE_NAME,'w+') as fh:
        # Pass the file handle in as a lambda function to make it callable
        model.summary(print_fn=lambda x: fh.write(x + '\n'))

        fh.write('\n---- Model evaluation ---\n')
        scores = model.evaluate(test_ds)
        fh.write(f'error on the test set : {scores[0]} | accuracy on the test set : {scores[1]}')

except:
    print("error during the printing process")