"""Contains losses used for performing semantics-aware image domain adaptation."""
'''The code is based on the loss functions of tensorflow implement of CycleGAN. 
The original code gives the formulation of Least-square GAN for computing 
the Adversarial loss and cycle-consistent loss for enforcing cycle-consistency.

SemGAN uses the LsGAN for Adversarial loss as well. Additionally, a classification
loss is added into the cycle-consistent loss part. Besides, specific losses for 
performing object transfiguration and domain adaptation tasks are formulated.'''

import tensorflow as tf
import keras
import model
import numpy
import tensorflow.keras.backend as K

eps = 1e-5

def categorical_crossentropy(y_true, y_pred):
    y_pred = K.clip(y_pred, eps, 1 - eps)
    return tf.abs(-K.mean(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred)))


def cycle_consistency_loss(real_images, generated_images, real_labels, generated_labels):
    """Compute the cycle consistency loss.

    The cycle consistency loss is defined as the sum of the L1 distances
    between the real images from each domain and their generated (fake)
    counterparts.

    This definition is derived from Equation 2 in:
        Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial
        Networks.
        Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros.


    Args:
        real_images: A batch of images from domain X, a `Tensor` of shape
            [batch_size, height, width, channels].
        generated_images: A batch of generated images made to look like they
            came from domain X, a `Tensor` of shape
            [batch_size, height, width, channels].
        real_labels: the one-hot encoded semantic labels from domain X. A tensor of shape
            [batch_size, height, width, N_Classes].
        generated labels: A batch of generated semantic maps made to look like 
            they came from domain X. A tensor of shape
            [batch_size, height, width, N_Classes].

    Returns:
        The cycle consistency loss.

    The description above is the original cycle-consistent loss.
    In SemGAN, a standard cross-entropy loss is included for computing the classification loss.
    """
    cycle_consistent_loss = tf.reduce_mean(tf.abs(real_images - generated_images))
    
    classification_loss = categorical_crossentropy(real_labels, generated_labels)

    cycle_loss = cycle_consistent_loss + classification_loss

    return cycle_loss

def lsgan_loss_generator(prob_fake_is_real):
    """Computes the LS-GAN loss as minimized by the generator.

    Rather than compute the negative loglikelihood, a least-squares loss is
    used to optimize the discriminators as per Equation 2 in:
        Least Squares Generative Adversarial Networks
        Xudong Mao, Qing Li, Haoran Xie, Raymond Y.K. Lau, Zhen Wang, and
        Stephen Paul Smolley.
        https://arxiv.org/pdf/1611.04076.pdf

    Args:
        prob_fake_is_real: The discriminator's estimate that generated images
            made to look like real images are real.

    Returns:
        The total LS-GAN loss.
    """
    return tf.reduce_mean(tf.squared_difference(prob_fake_is_real, 1))


def lsgan_loss_discriminator(prob_real_is_real, prob_fake_is_real):
    """Computes the LS-GAN loss as minimized by the discriminator.

    Rather than compute the negative loglikelihood, a least-squares loss is
    used to optimize the discriminators as per Equation 2 in:
        Least Squares Generative Adversarial Networks
        Xudong Mao, Qing Li, Haoran Xie, Raymond Y.K. Lau, Zhen Wang, and
        Stephen Paul Smolley.
        https://arxiv.org/pdf/1611.04076.pdf

    Args:
        prob_real_is_real: The discriminator's estimate that images actually
            drawn from the real domain are in fact real.
        prob_fake_is_real: The discriminator's estimate that generated images
            made to look like real images are real.

    Returns:
        The total LS-GAN loss.
    """
    return (tf.reduce_mean(tf.squared_difference(prob_real_is_real, 1)) +
            tf.reduce_mean(tf.squared_difference(prob_fake_is_real, 0))) * 0.5


def semantic_consistency_loss(real_labels, generated_labels):
    '''For domain adaptation, it is essential to preserve the semantic informatiion.
    SemGAN adds a cross-domain semantic consistency loss for this purpose.

    The cross-domain semantic consistency loss is formulated from equation 6 in:
        Semantics-Aware Image to Image Translation and Domain Transfer.
        Pravakar Roy, Nicolai Häni, Volkan Isler.

    Args:
        real_labels: the one-hot encoded semantic labels from domain X. A tensor of shape
            [batch_size, height, width, N_Classes].
        generated_labels: the generated semantic labels in another domain Y. A tensor of shape
            [batch_size, height, width, N_Classes].
        

    Returns:
        The cross-domain semantic consistency loss.
    '''

    '''real_labels = tf.reshape(real_labels, [-1, model.N_CLASSES])
    generated_labels = tf.reshape(generated_labels, [-1, model.N_CLASSES])

    cce_losses = tf.keras.losses.categorical_crossentropy(real_labels, generated_labels)
    sem_consistent_loss = tf.abs(tf.keras.backend.mean(cce_losses))'''

    sem_consistent_loss = categorical_crossentropy(real_labels, generated_labels)
    

    return sem_consistent_loss





