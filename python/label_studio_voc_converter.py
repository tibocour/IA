import argparse
import os
import glob
import tempfile
import shutil
import random
import xml.etree.ElementTree as ET
import imgaug as ia
import cv2
from pascal_voc_writer import Writer


def parse_xml(filename):
    tree = ET.parse(filename)
    elem = tree.getroot()
    result = {
        'filename': elem.find('.//filename').text,
        'size': {
            'width': elem.find('.//size/width').text,
            'height': elem.find('.//size/height').text,
            'depth': elem.find('.//size/depth').text,
        },
        'objects': []
    }

    for e in elem.findall('.//object'):
        obj = {
            'name': e.find('.//name').text,
            'xmin': e.find('.//bndbox/xmin').text,
            'ymin': e.find('.//bndbox/ymin').text,
            'xmax': e.find('.//bndbox/xmax').text,
            'ymax': e.find('.//bndbox/ymax').text
        }
        result['objects'].append(obj)

    return result


def augmentation_pipeline():
    def sometimes(aug): 
        return ia.augmenters.Sometimes(0.5, aug)

    return ia.augmenters.Sequential(
        [
            # apply the following augmenters to most images
            ia.augmenters.Fliplr(0.5),  # horizontally flip 50% of all images
            #ia.augmenters.Flipud(0.5),  # vertically flip 20% of all images
            # crop images by -2% to 5% of their height/width
            sometimes(ia.augmenters.CropAndPad(
                percent=(-0.02, 0.05),
                pad_mode=ia.ALL,
                pad_cval=(0, 255)
            )),
            sometimes(ia.augmenters.Affine(
                # scale images to 95-105% of their size, individually per axis
                scale={'x': (0.95, 1.05), 'y': (0.95, 1.05)},
                # translate by -5 to +5 percent (per axis)
                translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)},
                rotate=(-5, 5),  # rotate by -5 to +5 degrees
                shear=(-2, 2),  # shear by -2 to +2 degrees
                # use nearest neighbour or bilinear interpolation (fast)
                order=[0, 1],
                # if mode is constant, use a cval between 0 and 255
                cval=(0, 255),
                # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                mode=ia.ALL
            )),
        ],
        random_order=True
    )


def augment(annotation, input_image_dir, output_image_dir, output_annotation_dir, size, file_ext):

    # create the augmentation pipeline
    pipeline = augmentation_pipeline()

    # filename from label studio : base filename (w/o extension neither path)
    filename = annotation['filename']

    # extension
    if filename not in file_ext:
      raise Exception(f"file {filename} is unknown...")

    ext = file_ext[filename]

    # load the original image
    original_image = cv2.imread(os.path.join(input_image_dir, f"{filename}{ext}"))

    # write the image in the output directory
    cv2.imwrite(os.path.join(output_image_dir, f"{filename}{ext}"), original_image)

    # create the original bboxes
    _bbs = [
        ia.BoundingBox(x1=int(obj["xmin"]),
                       y1=int(obj["ymin"]),
                       x2=int(obj["xmax"]),
                       y2=int(obj["ymax"]),
                       label=obj["name"])
        for obj in annotation['objects']
    ]
    original_bbs = ia.BoundingBoxesOnImage(_bbs, shape=original_image.shape)

    # save the bboxes in the output directory
    writer = Writer(path=os.path.join("images", f"{filename}{ext}"),
                    width=annotation['size']['width'],
                    height=annotation['size']['height'],
                    database="BFC")
    for bb in original_bbs.bounding_boxes:
        if int((bb.x2 - bb.x1) * (bb.y2 - bb.y1)) == 0:
            print("original boundingbox has non existing area. Skipping")
            continue
        writer.addObject(bb.label,
                         int(bb.x1),
                         int(bb.y1),
                         int(bb.x2),
                         int(bb.y2))
    writer.save(os.path.join(output_annotation_dir, f"{filename}.xml"))

    # loop over the number of augmentation
    for i in range(size):

        # augmentation should be reproductible (img and bbox)
        pipeline_ = pipeline.to_deterministic()

        # augment the image
        image_aug = pipeline_.augment_images([original_image])[0]
        
        # save the image in the output directory
        cv2.imwrite(os.path.join(output_image_dir, f"{filename}_{i}{ext}"), image_aug)
        
        # augment the bboxes
        bbs_aug = pipeline_.augment_bounding_boxes([original_bbs])[0].remove_out_of_image().clip_out_of_image()
        # save the bboxes in the output directory
        writer = Writer(path=os.path.join(output_image_dir, f"{filename}_{i}{ext}"),
                        width=annotation['size']['width'],
                        height=annotation['size']['height'],
                        database="BFC")
        for bb in bbs_aug.bounding_boxes:
            if int((bb.x2-bb.x1)*(bb.y2-bb.y1)) == 0:
                print("augmented boundingbox has non existing area. Skipping")
                continue
            writer.addObject(bb.label,
                             int(bb.x1),
                             int(bb.y1),
                             int(bb.x2),
                             int(bb.y2))
        writer.save(os.path.join(output_annotation_dir, f"{filename}_{i}.xml"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Pascal VOC Label Studio Converter for BirdsForChange")
    parser.add_argument("--zip", type=str, required=True)
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--size", type=int, default=10)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as zip_tmpdirname:
        print(f"created zip temporary directory: {zip_tmpdirname}")

        print(f"extract zip file {args.zip} in {zip_tmpdirname}")
        shutil.unpack_archive(filename=args.zip, extract_dir=zip_tmpdirname, format="zip")

        annotation_dir = os.path.join(zip_tmpdirname, "Annotations")
        image_dir = os.path.join(zip_tmpdirname, "images")

        # check whether the Pascal VOC xml Dataset seems valid
        for path in [annotation_dir, image_dir]:
            if not os.path.exists(path):
                raise FileExistsError(path)
            if not os.path.isdir(path):
                raise NotADirectoryError(path)

        xml_files = glob.glob(f"{annotation_dir}/*.xml")

        print(f" * number of xml files : {len(xml_files)}")

        train_size = int(len(xml_files) * args.train_split)

        print(f" * train size : {train_size}")
        print(f" * valid size : {len(xml_files) - train_size}")

        # random maybe useless...
        random.shuffle(xml_files)

        infos = {
            "train": {
                "xml_files": xml_files[:train_size],
                "size": args.size,
            },
            "valid": {
                "xml_files": xml_files[train_size:],
                "size": 0,
            }
        }

        for mode, params in infos.items():

            with tempfile.TemporaryDirectory() as dataset_tmpdirname:
                print(f"created dataset temporary directory: {dataset_tmpdirname}")
                print(f"create annotations and images directories")
                dataset_image_dir = os.path.join(dataset_tmpdirname, "images")
                os.makedirs(dataset_image_dir, exist_ok=False)
                dataset_annotation_dir = os.path.join(dataset_tmpdirname, "annotations")
                os.makedirs(dataset_annotation_dir, exist_ok=False)

                print(f"create dataset {mode} with params {params}")

                print("create file / extension mapping")
                file_ext = os.listdir(image_dir)
                file_ext = [os.path.splitext(f) for f in file_ext]
                file_ext = {f:e for f, e in file_ext}

                for file in params["xml_files"]:

                    # get the annotation from Pascal VOC xml file
                    ann = parse_xml(file)

                    # do the augmentation
                    augment(annotation=ann,
                            input_image_dir=image_dir,
                            output_image_dir=dataset_image_dir,
                            output_annotation_dir=dataset_annotation_dir,
                            size=params["size"],
                            file_ext=file_ext)

                dataset_zipfile = f"{mode}_{os.path.basename(args.zip)}"
                print(f"created zip file: {dataset_zipfile}")
                shutil.make_archive(base_name=dataset_zipfile.replace(".zip", ""),
                                    format="zip",
                                    root_dir=dataset_tmpdirname)
