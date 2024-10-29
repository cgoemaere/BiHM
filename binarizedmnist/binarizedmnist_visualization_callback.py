import matplotlib.pyplot as plt

from .visualization_callback import VisualizationCallback


class BinarizedMNISTVisualizationCallback(VisualizationCallback):
    viz_order = ("img",)

    def visualize_img(self, img, img_pred):
        # If no prediction, just show input img
        if img_pred is None:
            img_pred = img

        # Show generated image
        plt.imshow(img_pred.view(28, 28).T.detach(), cmap="gray")
