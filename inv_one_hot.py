import tensorflow as tf
import model
import numpy as np

palette = np.array(
[[128,  64, 128],
 [244,  35, 232],
 [70,   70,  70],
 [102, 102, 156],
 [190, 153, 153],
 [153, 153, 153],
 [250, 170,  30],
 [220, 220,   0],
 [107, 142,  35],
 [152, 251, 152],
 [ 70, 130, 180],
 [220,  20,  60],
 [255,   0,   0],
 [  0,   0, 142],
 [  0,   0,  70],
 [  0,  60, 100],
 [  0,  80, 100],
 [  0,   0, 230],
 [119,  11,  32]], np.uint8)

palette = tf.constant(palette, dtype=tf.uint8)

def back_img(gen_map, palette):

	class_indexes = tf.argmax(gen_map, axis=-1)
	# This operation flattens class_indexes
	class_indexes = tf.reshape(class_indexes, [-1])
	color_image = tf.gather(palette, class_indexes)
	color_image = tf.reshape(color_image, 
		[1, model.IMG_HEIGHT, model.IMG_WIDTH, model.IMG_CHANNELS])
	color_image = tf.cast(color_image, tf.float32)

	return color_image