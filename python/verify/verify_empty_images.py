import os
import shutil
from xml.etree import ElementTree as ET

# Create a folder to store unannotated images
if not os.path.exists('Unannotated_Images'):
    os.makedirs('Unannotated_Images')

# Obtenir les noms de fichiers sans extension dans les dossiers "Images" et "Annotations"
image_files = [os.path.splitext(filename)[0] for filename in os.listdir('Images')]
annotation_files = [os.path.splitext(filename)[0] for filename in os.listdir('Annotations')]

# Copier les images sans annotations dans un dossier "Unannotated_Images"
for image_file in image_files:
    if image_file not in annotation_files:
        # Copier l'image dans le dossier "Unannotated_Images"
        shutil.copy(f'Images/{image_file}.jpg', f'Unannotated_Images/{image_file}.jpg')
        print(f'{image_file}.jpg has been copied to Unannotated_Images folder.')
        