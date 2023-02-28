import os

# Obtenir les noms de fichiers sans extension dans les dossiers "Images" et "Annotations"
image_files = [os.path.splitext(filename)[0] for filename in os.listdir('images')]
annotation_files = [os.path.splitext(filename)[0] for filename in os.listdir('Annotations')]
        
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
