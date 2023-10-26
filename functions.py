from model import *
import PIL
import matplotlib.pyplot as plt
import cv2
from matplotlib.colors import ListedColormap
import os

dir = os.getcwd()

def predict(image_path):
    """
    Prédit la segmentation d'une image à l'aide d'un modèle U-Net pré-entraîné.

    :param image_path: Chemin vers l'image à segmenter.
    :return: Chemin vers l'image de prédiction.
    """
    path_to_load = os.path.join(dir, "model.h5")

    model = build_unet(input_shape=(512, 512, 3),
                        filters=[2 ** i for i in range(5, int(np.log2(2048) + 1))],
                        batchnorm=False, transpose=False, dropout_flag=False)

    model.load_weights(path_to_load)

    sample_image = np.asarray(PIL.Image.open(image_path))
    prediction = model.predict(sample_image[tf.newaxis, ...])[0]

    prediction_class1 = np.copy(prediction[..., 0])
    prediction_class2 = np.copy(prediction[..., 1])
    prediction[..., 0] = prediction_class2
    prediction[..., 1] = prediction_class1
    prediction_path = os.path.join(dir, "prediction.png")
    
    plt.imsave(prediction_path, prediction)

    return prediction_path


def compute_percentage(image_path):
    """
    Calcule le pourcentage de pixels verts dans une image.

    :param image_path: Chemin vers l'image à analyser.
    :return: Pourcentage de pixels verts.
    """
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    green_lower = np.array([0, 128, 0], dtype=np.uint8)
    green_upper = np.array([100, 255, 100], dtype=np.uint8)
    green_mask = cv2.inRange(image_rgb, green_lower, green_upper)

    green_pixel_count = np.count_nonzero(green_mask)
    total_pixel_count = image.shape[0] * image.shape[1]
    percentage_green = (green_pixel_count / total_pixel_count) * 100

    return percentage_green


def analyse(image_path):
    """
    Analyse une image, génère un graphique et estime l'absorption de carbone et le nombre d'arbres.

    :param image_path: Chemin vers l'image à analyser.
    :return: Chemin vers le graphique, estimation de l'absorption de carbone, nombre d'arbres.
    """
    figure_path = os.path.join(dir, "figure.png")
    percentage_green = compute_percentage(image_path)
    size = [percentage_green / 100, (100 - percentage_green) / 100]
    percents = [str(round(i*100, 2))+"%" for i in size]
    names = [f"Forest: {percents[0]}", f"Other: {percents[1]}"]
    my_circle = plt.Circle((0, 0), 0.6, color='white')
    custom_colors = [(196/255, 135/255, 81/255), (65/255, 166/255, 28/255)]

    if size[0] <= .5:
        custom_colors = [(65/255, 166/255, 28/255), (196/255, 135/255, 81/255)]

    cmap = ListedColormap(custom_colors)
    
    plt.figure(figsize=(10, 10))
    plt.title("Percentage of forest area", fontsize=27)
    plt.pie(size, labels=names, colors=cmap(size), textprops={'fontsize': 16})
    p = plt.gcf()
    p.gca().add_artist(my_circle)
    plt.savefig(figure_path)
    
    total_pixels = 512**2
    surface_verte_pixels = (percentage_green / 100) * total_pixels
    surface_verte_metres_carres = surface_verte_pixels / 2.25

    rand = random.randrange(25000, 35001)
    absorption_carbone_kg_an = round((rand * surface_verte_metres_carres) / 10000, 2)

    tree_area = (random.randint(10, 13) / 2) * np.pi
    trees_number = int(surface_verte_metres_carres / tree_area)

    return figure_path, absorption_carbone_kg_an, trees_number