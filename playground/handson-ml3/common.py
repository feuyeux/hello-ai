# Where to save the figures
import platform
from pathlib import Path

from matplotlib import pyplot as plt


def dataset_path():
    if platform.system() == "Windows":
        return Path("d:/park/handson-ml3/datasets")
    elif platform.system() == "Linux" or platform.system() == "Darwin":
        return Path.home() / "handson-ml3/datasets"
    else:
        print("Unsupported operating system")
        return None


def image_path():
    if platform.system() == "Windows":
        return Path("d:/park/handson-ml3/images/fundamentals")
    elif platform.system() == "Linux" or platform.system() == "Darwin":
        return Path.home() / "handson-ml3/images/fundamentals"
    else:
        print("Unsupported operating system")
        return None


def save_fig(fig_id, image_path, tight_layout=True, fig_extension="png", resolution=300):
    path = image_path / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


def plot_digit(image_data):
    image = image_data.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")
