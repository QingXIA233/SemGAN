"""Code for constructing the model and get the outputs from the model."""

import tensorflow as tf
import layers
import loader_one_hot
import inv_one_hot

# The number of samples per batch.
BATCH_SIZE = 1

# The height of each image.
IMG_HEIGHT = 256

# The width of each image.
IMG_WIDTH = 256

# The number of color channels per image.
IMG_CHANNELS = 3

# The number of classes in both domains
N_CLASSES = 19

# The number of filters in the encoder
nef = 64

ngf = 32
ndf = 64

POOL_SIZE = 50


def get_outputs(inputs, network="tensorflow"):
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
        current_generator_img = decoder_img
        current_generator_map = decoder_map

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

        fake_images_b = current_generator_img(stack_input_a, name="g_A")
        fake_images_a = current_generator_img(stack_input_b, name="g_B")

        fake_oh_labels_b = current_generator_map(stack_input_a, name="g_A")
        fake_oh_labels_a = current_generator_map(stack_input_b, name="g_B")

        stack_fake_a = tf.concat([fake_images_a, fake_oh_labels_a], axis=3) 
        stack_fake_b = tf.concat([fake_images_b, fake_oh_labels_b], axis=3) 

        fake_labels_a = inv_one_hot.back_img(fake_oh_labels_a, palette)
        fake_labels_b = inv_one_hot.back_img(fake_oh_labels_b, palette)

        scope.reuse_variables()

        prob_fake_a_is_real = current_discriminator(stack_fake_a, "d_A")
        prob_fake_b_is_real = current_discriminator(stack_fake_b, "d_B")

        cycle_images_a = current_generator_map(stack_fake_b, "g_B", skip=skip)
        cycle_images_b = current_generator_map(stack_fake_a, "g_A", skip=skip)

        cycle_oh_labels_b = current_generator_map(stack_fake_a, name="g_A")
        cycle_oh_labels_a = current_generator_map(stack_fake_b, name="g_B")

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
            out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c1")
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = layers.general_conv2d(
            out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c2", do_relu=False)

        return tf.nn.relu(out_res + inputres)

def encoder(input_enc, name="encoder"):
    """Build an encoder which encodes the stacked image and label (Bs, H, W, 4)
    into  a feature map of size (Bs, H/4, W/4, 256)"""
    with tf.variable_scope(name):
        f = 7
        ks = 3
        padding = "CONSTANT"

        pad_input = tf.pad(input_enc, [[0, 0], [ks, ks], [
            ks, ks], [0, 0]], padding)
        #pad_input.shape = (1, 262, 262, 3)

        o_c1 = layers.general_conv2d(
            pad_input, nef, f, f, 1, 1, 0.02, name="c1")
        #o_c1.shape = (1, 256, 256, 64)

        o_c2 = layers.general_conv2d(
            o_c1, nef * 2, ks, ks, 2, 2, 0.02, "SAME", "c2")
        #o_c2.shape = (1, 128, 128, 128)

        o_c3 = layers.general_conv2d(
            o_c2, nef * 4, ks, ks, 2, 2, 0.02, "SAME", "c3")
        #o_c3.shape = (1, 64, 64, 256)

        o_r1 = build_resnet_block(o_c3, nef * 4, "r1", padding)
        o_r2 = build_resnet_block(o_r1, nef * 4, "r2", padding)
        o_r3 = build_resnet_block(o_r2, nef * 4, "r3", padding)
        o_r4 = build_resnet_block(o_r3, nef * 4, "r4", padding)
        o_r5 = build_resnet_block(o_r4, nef * 4, "r5", padding)
        o_r6 = build_resnet_block(o_r5, nef * 4, "r6", padding)
        o_r7 = build_resnet_block(o_r6, nef * 4, "r7", padding)
        o_r8 = build_resnet_block(o_r7, nef * 4, "r8", padding)
        o_r9 = build_resnet_block(o_r8, nef * 4, "r9", padding)
        #o_r9.shape = (1, 64, 64, 256)

        return o_r9

def decoder_img(input_dec, name="img_generator"):
    """Build a decoder which decodes the output of the encoder to 
    generate images, it contains several upsampling(deconv) layers
    to generate images of shape(1, 256, 256, 3)"""
    with tf.variable_scope(name):
        f = 7
        ks = 3
        padding = "CONSTANT"

        input_dec = encoder(input_dec, name="enc1")

        o_d1 = layers.general_deconv2d(
            input_dec, [BATCH_SIZE, 128, 128, nef * 2], nef * 2, ks, ks, 2, 2, 0.02,
            "SAME", "d1")
        #o_d1.shape = (1, 128, 128, 128)

        o_d2 = layers.general_deconv2d(
            o_d1, [BATCH_SIZE, 256, 256, nef], nef, ks, ks, 2, 2, 0.02,
            "SAME", "d2")
        #o_d2.shape = (1, 256, 256, 64)

        o_c1 = layers.general_conv2d(o_d2, IMG_CHANNELS, f, f, 1, 1,
                                     0.02, "SAME", "c1",
                                     do_norm=False, do_relu=False)
        #o_c1.shape = (1, 256, 256, 3)

        out_dec_img = tf.nn.tanh(o_c1, "t1")

        return out_dec_img

def decoder_map(input_dec, name="map_generator"):
    """Build a decoder which decodes the output of the encoder to 
    generate one-hot encoded mask, it contains several upsampling(deconv) 
    layers to generate masks of shape(1, 256, 256, 19)"""
    with tf.variable_scope(name):
        f = 7
        ks = 3
        padding = "CONSTANT"

        size_d1 = input_dec.get_shape().as_list()
        o_d1 = layers.upsamplingDeconv(input_dec, size=[size_d1[1] * 2, size_d1[2] * 2], is_scale=False, method=1,
                                   align_corners=False,name= 'up1')
        #o_d1.shape = (1, 128, 128, 256)

        o_d1_end = layers.general_conv2d(o_d1, nef * 2 , ks, ks, 1, 1, 0.02, padding='same', name='d1')
        #o_d1_end.shape = (1, 128, 128, 128)

        size_d2 = o_d1_end.get_shape().as_list()
        o_d2 = layers.upsamplingDeconv(o_d1_end, size=[size_d2[1] * 2, size_d2[2] * 2], is_scale=False, method=1,
                                       align_corners=False, name='up2')
        #o_d2.shape = (1, 256, 256, 128)

        o_d2_end = layers.general_conv2d(o_d2, nef, ks, ks, 1, 1, 0.02, padding='same', name='d2')
        #o_d2_end.shape = (1, 256, 256, 64)

        o_c1 = layers.general_conv2d(o_d2_end, N_CLASSES, f, f, 1, 1,
                                     0.02, "SAME", "c1",
                                     do_norm=False, do_relu=False)
        #o_c1.shape = (1, 256, 256, 19)

        #Use softmax activation 

        out_dec_map = tf.nn.softmax(o_c1, "softmax") 


def patch_discriminator(inputdisc, name="discriminator"):
    with tf.variable_scope(name):
        f = 4

        patch_input = tf.random_crop(inputdisc, [1, 70, 70, 22])
        o_c1 = layers.general_conv2d(patch_input, ndf, f, f, 2, 2,
                                     0.02, "SAME", "c1", do_norm="False",
                                     relufactor=0.2)
        o_c2 = layers.general_conv2d(o_c1, ndf * 2, f, f, 2, 2,
                                     0.02, "SAME", "c2", relufactor=0.2)
        o_c3 = layers.general_conv2d(o_c2, ndf * 4, f, f, 2, 2,
                                     0.02, "SAME", "c3", relufactor=0.2)
        o_c4 = layers.general_conv2d(o_c3, ndf * 8, f, f, 2, 2,
                                     0.02, "SAME", "c4", relufactor=0.2)
        o_c5 = layers.general_conv2d(
            o_c4, 1, f, f, 1, 1, 0.02, "SAME", "c5", do_norm=False,
            do_relu=False)

        return o_c5
