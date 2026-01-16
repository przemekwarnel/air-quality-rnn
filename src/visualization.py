import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(history, metric):
    """
    Plots training and validation loss alongside another metric over epochs.

    Args:
        history (dict): A dictionary containing training history.
        metric (str): The name of the metric to plot (other than loss, e.g.'accuracy').
    """
    fig, ax1 = plt.subplots()#figsize=(8,6))
    ax2 = ax1.twinx()
    l1, = ax1.plot(history['loss'], '-', color='C0', label='loss')
    l2, = ax1.plot(history['val_loss'], '-', color='C1', label='val_loss')
    l3, = ax2.plot(history[metric], '-', color='C2', label=metric)
    l4, = ax2.plot(history[f'val_{metric}'], '', color='C3', label=f'val_{metric}')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel('loss')
    ax2.set_ylabel(metric)
    plt.legend(handles=[l1, l2, l3, l4], loc='center right')
    plt.show()


def visualize_imgs_preds(images, labels, preds, class_names, n_img):
    """
    Visualizes a few images with their true and predicted labels.

    Args:
        images (numpy array): Array of input images.
        labels (numpy array): Array of true labels.
        preds (numpy array): Array of predicted values.
        class_names (list): List of class names corresponding to label indices.
        n_img (int): Number of images to visualize.
    """
    # Prepare predictions and labels
    ## If binary classes (0/1)
    if preds.shape[1] == 1 or preds.ndim == 1:  
        preds = (preds > 0.5).astype(int).flatten()
        labels = labels.astype(int).flatten()
    else:
    ## For multi-class classification
        preds = np.argmax(preds, axis=1)
        labels = np.argmax(labels, axis=1)
    
    # Plot randomly selected images with true and predicted labels
    indices = np.random.choice(len(images), n_img, replace=False)
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(indices):
        plt.subplot(1, n_img, i + 1)
        plt.imshow(images[idx])
        plt.title(f"True: {class_names[labels[idx]]}\nPred: {class_names[preds[idx]]}")
        plt.axis('off')
    plt.show()

