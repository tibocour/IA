import os
import xml.etree.ElementTree as ET
import os

# Obtenir les noms de fichiers sans extension dans les dossiers "Images" et "Annotations"
image_files = [os.path.splitext(filename)[0] for filename in os.listdir('Unannotated_Images')]
annotation_files = [os.path.splitext(filename)[0] for filename in os.listdir('Annotations')]

# Vérifier les correspondances entre les noms de fichiers
for image_file in image_files:
    if image_file not in annotation_files:
        # Si le fichier n'a pas de fichier correspondant dans le dossier "Annotations", créer un nouveau fichier d'annotation
        root = ET.Element("annotation")
        folder = ET.SubElement(root, "folder")
        folder.text = "Images"
        filename = ET.SubElement(root, "filename")
        filename.text = f"{image_file}"
        path = ET.SubElement(root, "path")
        path.text = f"Images/{image_file}.jpg"
        source = ET.SubElement(root, "source")
        database = ET.SubElement(source, "database")
        database.text = "roboflow.ai"
        size = ET.SubElement(root, "size")
        width = ET.SubElement(size, "width")
        width.text = "640"
        height = ET.SubElement(size, "height")
        height.text = "480"
        depth = ET.SubElement(size, "depth")
        depth.text = "3"
        segmented = ET.SubElement(root, "segmented")
        segmented.text = "0"
        obj = ET.SubElement(root, "object")
        obj_name = ET.SubElement(obj, "name")
        obj_name.text = "vide"
        obj_pose = ET.SubElement(obj, "pose")
        obj_pose.text = "Unspecified"
        obj_truncated = ET.SubElement(obj, "truncated")
        obj_truncated.text = "0"
        obj_truncated = ET.SubElement(obj, "truncated")
        obj_truncated.text = "0"
        obj_difficult = ET.SubElement(obj, "difficult")
        obj_difficult.text = "0"
        obj_occluded = ET.SubElement(obj, "occluded")
        obj_occluded.text = "0"
        bndbox = ET.SubElement(obj, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = "0"
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = "0"
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = "640"
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = "480"

        tree = ET.ElementTree(root)
        tree.write(f"Annotations/{image_file}.xml")
        print(f"Created Annotations/{image_file}.xml because it did not have a corresponding annotation file.")
        
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