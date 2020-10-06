import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import pathlib
from sklearn.metrics import confusion_matrix
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction = 0.33)
sess = tf.compat.v1.Session(config = tf.compat.v1.ConfigProto(gpu_options = gpu_options))
with tf.device('/gpu:0'):
	train_data_dir  = "Mask_Datasets/Train"
	val_data_dir = "Mask_Datasets/Validation"
	train_data_dir = pathlib.Path(train_data_dir)
	val_data_dir = pathlib.Path(val_data_dir)

	CLASS_NAMES = np.array([item.name for item in train_data_dir.glob('*')])
	print(CLASS_NAMES)
	image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

	BATCH_SIZE = 32
	IMG_HEIGHT = 224
	IMG_WIDTH = 224

	train_data_gen = image_generator.flow_from_directory(directory = str(train_data_dir),
														 batch_size = 2,
														 shuffle = False,
														 target_size = (IMG_HEIGHT,IMG_WIDTH),
														 classes = list(CLASS_NAMES),
														)

	validation_data_gen = image_generator.flow_from_directory(directory = str(val_data_dir),
														 batch_size = 2,
														 shuffle = False,
														 target_size = (IMG_HEIGHT,IMG_WIDTH),
														 classes = list(CLASS_NAMES),
														)

	img_batch , label_batch = next(train_data_gen)
	img_shape = img_batch[0].shape
	# print(img_batch[0].shape)

	# plt.imshow(img_batch[0])
	# plt.axis("off")
	# plt.show()
	# print(label_batch[0])

	base_model = tf.keras.applications.VGG16(input_shape=img_shape,
	                                         include_top=False,
	                                         weights='imagenet')

	base_model.trainable = False
	feature_batch = base_model(img_batch)
	global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
	feature_batch_average = global_average_layer(feature_batch)
	# dense1 = tf.keras.layers.Dense(1024, activation = 'relu')
	# preprediction_batch = dense1(feature_batch_average)
	prediction_layer = tf.keras.layers.Dense(2, activation = "softmax")
	preprediction_batch = prediction_layer(feature_batch_average)
	'''
	model = tf.keras.Sequential([
			base_model,
			global_average_layer,
			#dense1,
			#tf.keras.layers.Dropout(0.3),
			prediction_layer
			])
	'''
	model = tf.keras.models.load_model("finalmodelvggnetindian2.h5")
	print(model.summary())
	nb_epochs = 20
	checkpoint = tf.keras.callbacks.ModelCheckpoint('finalmodelvggnetindian3.h5', verbose=1, monitor='val_accuracy',save_best_only=True, mode='auto')
	model.compile(optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01),loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),metrics=['accuracy'])


	history = model.fit(train_data_gen,
                    epochs=nb_epochs,
                    validation_data = validation_data_gen,
                    callbacks = [checkpoint])
	'''
	
	val_data_dir = "Mask_Datasets/Validation"
	val_data_dir = pathlib.Path(val_data_dir)
	validation_data_gen = image_generator.flow_from_directory(directory = str(val_data_dir),
														 batch_size = 8,
														 shuffle = False,
														 target_size = (IMG_HEIGHT,IMG_WIDTH),
														 classes = list(CLASS_NAMES),
														)

	model = tf.keras.models.load_model("finalmodel.h5")
	predictions = model.predict_generator(validation_data_gen)
	print(np.argmax(predictions,axis = 1))
	# y_pred = np.argmax(np.array(predictions),axis = 1)
	# y_true = np.array(validation_data_gen.classes)
	# print(confusion_matrix(y_true,y_pred))
	'''