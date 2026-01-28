import matplotlib.pyplot as plt


def plot_loss_curves(history, metric='accuracy'):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  metric_for_plot = history.history[metric]
  val_metric_for_plot = history.history[f'val_{metric}']

  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, metric_for_plot, label='training_metric')
  plt.plot(epochs, val_metric_for_plot, label='val_metric')
  plt.title('Metric')
  plt.xlabel('Epochs')
  plt.legend();
