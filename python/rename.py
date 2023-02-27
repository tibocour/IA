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
        
# Renommer chaque paire de fichiers
image_counter = 1
annotation_counter = 1

for index, (image_file, annotation_file) in enumerate(zip(sorted(image_files), sorted(annotation_files)), start=1):
    # Renommer le fichier d'image
    os.rename(f'Images/{image_file}.jpg', f'Images/{index:04d}.jpg')
    
    # Renommer le fichier d'annotation
    os.rename(f'Annotations/{annotation_file}.xml', f'Annotations/{index:04d}.xml')
    
    image_counter += 1
    annotation_counter += 1

    # Modifier le fichier d'annotation
    os.system(f'xmlstarlet ed -L -u "(//annotation/filename)" -v "{index:04d}" -u "(//annotation/path)" -v "Images/{index:04d}.jpg" Annotations/{index:04d}.xml')
