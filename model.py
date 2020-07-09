"""Code for constructing the model and get the outputs from the model."""

import tensorflow as tf
import layers
import loader_one_hot
import inv_one_hot
import tensorflow.keras.backend as K

# The number of samples per batch.
BATCH_SIZE = 1

# The height of each image.
IMG_HEIGHT = 128

# The width of each image.
IMG_WIDTH = 128

# The number of color channels per image.
IMG_CHANNELS = 3

# The number of classes in both domains
N_CLASSES = 19

POOL_SIZE = 50

#The number of the filters in the first layer of generator
#and discriminator
ngf = 32
ndf = 64

def get_outputs(inputs, network="tensorflow", skip=False):
    images_a = inputs['images_a']
    images_b = inputs['images_b']
    labels_a = inputs['labels_a']
    labels_b = inputs['labels_b']

    fake_pool_a = inputs['fake_pool_a']
    fake_pool_b = inputs['fake_pool_b']
    fake_pool_a_label = inputs['fake_pool_a_label']
    fake_pool_b_label = inputs['fake_pool_b_label']

    with tf.variable_scope("Model") as scope:

    	current_discriminator = patch_discriminator
    	current_generator = generator_img_map

    	#convert the label maps to one-hot encoding
    	palette = loader_one_hot.palette

    	# Convert the RGB semantic map to one-hot encodings(1, 256, 256, 19)
    	oh_labels_a = loader_one_hot.one_hot(labels_a, palette)
    	oh_labels_b = loader_one_hot.one_hot(labels_b, palette)

    	# Concatenate the input image and mask into a tensor(1, 256, 256, 22)
    	stack_input_a = tf.concat([images_a, oh_labels_a], axis=3) 
    	stack_input_b = tf.concat([images_b, oh_labels_b], axis=3)

    	prob_real_a_is_real = current_discriminator(stack_input_a, "d_A")
    	prob_real_b_is_real = current_discriminator(stack_input_b, "d_B")

    	fake_images_b, fake_oh_labels_b = current_generator(stack_input_a, name="g_A")
    	fake_images_a, fake_oh_labels_a = current_generator(stack_input_b, name="g_B")

    	stack_fake_a = tf.concat([fake_images_a, fake_oh_labels_a], axis=3) 
    	stack_fake_b = tf.concat([fake_images_b, fake_oh_labels_b], axis=3)

    	scope.reuse_variables()

    	prob_fake_a_is_real = current_discriminator(stack_fake_a, "d_A")
    	prob_fake_b_is_real = current_discriminator(stack_fake_b, "d_B")

    	fake_labels_a = inv_one_hot.back_img(fake_oh_labels_a, palette)
    	fake_labels_b = inv_one_hot.back_img(fake_oh_labels_b, palette)

    	cycle_images_a, cycle_oh_labels_a = current_generator(stack_fake_b, "g_B")
    	cycle_images_b, cycle_oh_labels_b = current_generator(stack_fake_a, "g_A")

    	oh_fake_pool_a_label = loader_one_hot.one_hot(fake_pool_a_label, palette)
    	oh_fake_pool_b_label = loader_one_hot.one_hot(fake_pool_b_label, palette)

    	stack_fake_pool_a = tf.concat([fake_pool_a, oh_fake_pool_a_label], axis=3)
    	stack_fake_pool_b = tf.concat([fake_pool_b, oh_fake_pool_b_label], axis=3)

    	scope.reuse_variables()
    	prob_fake_pool_a_is_real = current_discriminator(stack_fake_pool_a, "d_A")
    	prob_fake_pool_b_is_real = current_discriminator(stack_fake_pool_b, "d_B")

    return {
        'prob_real_a_is_real': prob_real_a_is_real,
        'prob_real_b_is_real': prob_real_b_is_real,
        'prob_fake_a_is_real': prob_fake_a_is_real,
        'prob_fake_b_is_real': prob_fake_b_is_real,
        'prob_fake_pool_a_is_real': prob_fake_pool_a_is_real,
        'prob_fake_pool_b_is_real': prob_fake_pool_b_is_real,
        'cycle_images_a': cycle_images_a,
        'cycle_images_b': cycle_images_b,
        'fake_images_a': fake_images_a,
        'fake_images_b': fake_images_b,
        'fake_labels_a': fake_labels_a,
        'fake_labels_b': fake_labels_b,
        'oh_labels_a': oh_labels_a,
        'oh_labels_b': oh_labels_b,
        'fake_oh_labels_a': fake_oh_labels_a,
        'fake_oh_labels_b': fake_oh_labels_b,
        'cycle_oh_labels_a': cycle_oh_labels_a,
        'cycle_oh_labels_b': cycle_oh_labels_b
    }
        

def build_resnet_block(inputres, dim, name="resnet", padding="REFLECT"):
    """build a single block of resnet.

    :param inputres: inputres
    :param dim: dim
    :param name: name
    :param padding: for tensorflow version use REFLECT; for pytorch version use
     CONSTANT
    :return: a single block of resnet.
    """
    with tf.variable_scope(name):
        out_res = tf.pad(inputres, [[0, 0], [1, 1], [
            1, 1], [0, 0]], padding)
        out_res = layers.general_conv2d(
            out_res, tf.constant(True, dtype=bool), dim, 3, 3, 1, 1, 0.02, "VALID", "c1")
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = layers.general_conv2d(
            out_res, tf.constant(True, dtype=bool), dim, 3, 3, 1, 1, 0.02, "VALID", "c2", do_relu=False)

        return tf.nn.relu(out_res + inputres)

def softmax(input):
    label_dim = -1
    d = K.exp(input - K.max(input, axis=label_dim, keepdims=True))
    return d / K.sum(d, axis=label_dim, keepdims=True)

def generator_img_map(inputgen, name="generator", skip=False):
    with tf.variable_scope(name):
        f = 7
        ks = 3
        padding = "REFLECT"

        pad_input = tf.pad(inputgen, [[0, 0], [ks, ks], [
            ks, ks], [0, 0]], padding)
        #pad_input.shape = (1, 134, 134, 3)

        o_c1 = layers.general_conv2d(
            pad_input, tf.constant(True, dtype=bool), ngf, f, f, 1, 1, 0.02, name="c1")
        #o_c1.shape = (1, 128, 128, 32)
        o_c2 = layers.general_conv2d(
            o_c1, tf.constant(True, dtype=bool), ngf * 2, ks, ks, 2, 2, 0.02, "SAME", "c2")
        #o_c2.shape = (1, 64, 64, 64)
        o_c3 = layers.general_conv2d(
            o_c2, tf.constant(True, dtype=bool), ngf * 4, ks, ks, 2, 2, 0.02, "SAME", "c3")
        #o_c3.shape = (1, 32, 32, 128)

        o_r1 = build_resnet_block(o_c3, ngf * 4, "r1", padding)
        o_r2 = build_resnet_block(o_r1, ngf * 4, "r2", padding)
        o_r3 = build_resnet_block(o_r2, ngf * 4, "r3", padding)
        o_r4 = build_resnet_block(o_r3, ngf * 4, "r4", padding)
        o_r5 = build_resnet_block(o_r4, ngf * 4, "r5", padding)
        o_r6 = build_resnet_block(o_r5, ngf * 4, "r6", padding)
        o_r7 = build_resnet_block(o_r6, ngf * 4, "r7", padding)
        o_r8 = build_resnet_block(o_r7, ngf * 4, "r8", padding)
        o_r9 = build_resnet_block(o_r8, ngf * 4, "r9", padding)
        #o_r9.shape = (1, 32, 32, 128)
        #o_r9 is the output of the encoder. Feed it two decoders.

        """Following is the structure of the decoder for decoding 
        the encoded feature map to generate images."""
        o_c4 = layers.general_deconv2d(
            o_r9, [BATCH_SIZE, 64, 64, ngf * 2], ngf * 2, ks, ks, 2, 2, 0.02,
            "SAME", "c4")
        #o_c4.shape = (1, 64, 64, 64)
        o_c5 = layers.general_deconv2d(
            o_c4, [BATCH_SIZE, 128, 128, ngf], ngf, ks, ks, 2, 2, 0.02,
            "SAME", "c5")
        #o_c5.shape = (1, 128, 128, 32)
        o_c6 = layers.general_conv2d(
        	o_c5, tf.constant(False, dtype=bool), IMG_CHANNELS, f, f, 1, 1, 0.02, "SAME", "c6", do_relu=False)
        #o_c6.shape = (1, 128, 128, 3) 

        out_dec_img = tf.nn.tanh(o_c6, "t1")
        #out_gen is the output of this decoder, which is the image.

        """Following is the structure of the decoder for decoding 
        the encoded feature map to generate semantic labels."""
        size_d1 = o_r9.get_shape().as_list()
        o_d1 = layers.upsamplingDeconv(o_r9, size=[size_d1[1] * 2, size_d1[2] * 2], is_scale=False, method=1,
                                   align_corners=False,name= 'up1')
        #o_d1.shape = (1, 64, 64, 128)

        o_d1_end = layers.general_conv2d(o_d1, 
        	tf.constant(True, dtype=bool), ngf * 2, ks, ks, 1, 1, padding='same', name='d1')
        #o_d1_end.shape = (1, 64, 64, 64)

        size_d2 = o_d1_end.get_shape().as_list()
        o_d2 = layers.upsamplingDeconv(o_d1_end, size=[size_d2[1] * 2, size_d2[2] * 2], is_scale=False, method=1,
                                       align_corners=False, name='up2')
        #o_d2.shape = (1, 128, 128, 64)

        o_d2_end = layers.general_conv2d(o_d2, 
        	tf.constant(True, dtype=bool), ngf, ks, ks, 1, 1, 0.02, padding='same', name='d2')
        #o_d2_end.shape = (1, 128, 128, 32)

        o_c7 = layers.general_conv2d(
        	o_d2_end, tf.constant(False, dtype=bool), N_CLASSES, f, f, 1, 1, 0.02, padding='SAME', name="c7", do_relu=False)
        #o_c4.shape = (1, 128, 128, 19)

        out_dec_map = softmax(o_c7)
        #out_gen is the output of this decoder, which is the semantic map.

        return out_dec_img, out_dec_map

def patch_discriminator(inputdisc, name="discriminator"):
    with tf.variable_scope(name):
        f = 4

        patch_input = tf.random_crop(inputdisc, [1, 70, 70, 22])
        o_c1 = layers.general_conv2d(patch_input, tf.constant(True, dtype=bool), ndf, f, f, 2, 2,
                                     0.02, "SAME", "c1", relufactor=0.2)
        o_c2 = layers.general_conv2d(o_c1, tf.constant(True, dtype=bool), ndf * 2, f, f, 2, 2,
                                     0.02, "SAME", "c2", relufactor=0.2)
        o_c3 = layers.general_conv2d(o_c2, tf.constant(True, dtype=bool), ndf * 4, f, f, 2, 2,
                                     0.02, "SAME", "c3", relufactor=0.2)
        o_c4 = layers.general_conv2d(o_c3, tf.constant(True, dtype=bool), ndf * 8, f, f, 2, 2,
                                     0.02, "SAME", "c4", relufactor=0.2)
        o_c5 = layers.general_conv2d(
            o_c4, tf.constant(False, dtype=bool), 1, f, f, 1, 1, 0.02, "SAME", "c5", do_relu=False)

        return o_c5







