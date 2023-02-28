import cv2
import os
import xml.etree.ElementTree as ET

# Spécifiez le chemin vers les images et le dossier d'annotations
IMAGE_PATH = "images/"
ANNOTATION_PATH = "Annotations/"
OUTPUT_PATH = "images_processed"

# Créer le dossier de sortie s'il n'existe pas
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

# Parcourez chaque fichier d'annotation XML dans le dossier
for filename in os.listdir(ANNOTATION_PATH):
    if not filename.endswith('.xml'):
        continue

    # Chargez le fichier d'annotation XML
    xml_path = os.path.join(ANNOTATION_PATH, filename)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Chargez l'image correspondante
    image_filename = root.find('filename').text
    image_path = os.path.join(IMAGE_PATH, image_filename + ".jpg")
    image = cv2.imread(image_path)

    # Dessinez une bounding box sur l'image pour chaque objet annoté
    for obj in root.findall('object'):
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)
        label = obj.find('name').text

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Enregistrez l'image avec les bounding boxes dans le dossier de sortie
    output_filename = os.path.splitext(filename)[0] + ".jpg"
    output_path = os.path.join(OUTPUT_PATH, output_filename)
    cv2.imwrite(output_path, image)