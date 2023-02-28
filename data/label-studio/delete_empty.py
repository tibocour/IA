import os

# Obtenir les noms de fichiers sans extension dans les dossiers "Images" et "Annotations"
image_files = [os.path.splitext(filename)[0] for filename in os.listdir('images')]
annotation_files = [os.path.splitext(filename)[0] for filename in os.listdir('Annotations')]

# Vérifier les correspondances entre les noms de fichiers
for image_file in image_files:
    if image_file not in annotation_files:
        # Si le fichier n'a pas de fichier correspondant dans le dossier "Annotations", supprimer le fichier
        os.remove(f'Images/{image_file}.jpg')
        print(f'{image_file}.jpg has been deleted because it does not have a corresponding annotation file.')
        
for annotation_file in annotation_files:
    if annotation_file not in image_files:
        # Si le fichier n'a pas de fichier correspondant dans le dossier "Images", supprimer le fichier
        os.remove(f'Annotations/{annotation_file}.xml')
        print(f'{annotation_file}.xml has been deleted because it does not have a corresponding image file.')