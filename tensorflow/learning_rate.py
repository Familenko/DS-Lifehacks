import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


lr_schedule_callback = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10**(epoch / 20))


def define_learning_rate(epoch, 
  history, 
  epoch_step = 20, 
  start_lr = 1e-6, 
  stop_lr = 1e-2, 
  view = 10):

    lrs = start_lr * (10 ** (np.arange(epoch) / epoch_step))
    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.semilogx(lrs, history.history["loss"])
    plt.tick_params('both', length=10, width=1, which='both')
    plt.axis([start_lr, stop_lr, 0, view])