import tensorflow as tf
import numpy as np
import layers

palette = np.array(
[[  0,   0,   0],
 [128,  64, 128],
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

def one_hot(label, palette):
	label = tf.cast(label, tf.uint8)
	semantic_map = []
	for colour in palette:
		class_map = tf.reduce_all(tf.equal(label, colour), axis=-1)
		semantic_map.append(class_map)
	semantic_map = tf.stack(semantic_map, axis=-1)
	# NOTE cast to tf.float32 because most neural networks operate in float32.
	semantic_map = tf.cast(semantic_map, tf.float32)

	return semantic_map