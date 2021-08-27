import os
import glob
import argparse
import shutil
import tempfile

import xml.etree.ElementTree as ET

import tensorflow as tf

from tflite_model_maker import object_detector
from tflite_model_maker.config import ExportFormat


def autodetect_labels(annotation_dir):
    xml_files = glob.glob(f"{annotation_dir}/*.xml")

    labels = set()

    for filename in xml_files:
        tree = ET.parse(filename)
        elem = tree.getroot()
        for e in elem.findall('.//object'):
            labels.add(e.find('.//name').text)

    labels = list(labels)
    labels.sort()

    return labels


def extract_dataset(zipname, dirname):

    print(f"extract zip file {zipname} in {dirname}")
    shutil.unpack_archive(filename=zipname, extract_dir=dirname, format="zip")

    images_dir = os.path.join(dirname, "images")
    annotations_dir = os.path.join(dirname, "annotations")

    labels = autodetect_labels(annotations_dir)
    print(f"labels : {labels}")

    return object_detector.DataLoader.from_pascal_voc(
        images_dir=images_dir,
        annotations_dir=annotations_dir,
        label_map=labels
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Litter Trainer for BirdsForChange")
    parser.add_argument("--train_zip", type=str, required=True)
    parser.add_argument("--valid_zip", type=str, required=True)
    args = parser.parse_args()

    print(f"Tensorflow version : {tf.__version__}")

    with tempfile.TemporaryDirectory() as zip_tmpdirname:

        print(f"created zip temporary directory: {zip_tmpdirname}")

        train_dir = os.path.join(zip_tmpdirname, "train")
        valid_dir = os.path.join(zip_tmpdirname, "valid")

        os.makedirs(train_dir, exist_ok=False)
        os.makedirs(valid_dir, exist_ok=False)

        train_data = extract_dataset(zipname=args.train_zip, dirname=train_dir)
        valid_data = extract_dataset(zipname=args.valid_zip, dirname=valid_dir)

        print(f"* train dataset len : {len(train_data)}")
        print(f"* valid dataset len : {len(valid_data)}")

        # model efficientdet
        # https://arxiv.org/pdf/1911.09070.pdf
        # https://ai.googleblog.com/2020/04/efficientdet-towards-scalable-and.html
        spec = object_detector.EfficientDetLite0Spec()

        # train the model
        model = object_detector.create(train_data=train_data,
                                       model_spec=spec,
                                       validation_data=valid_data,
                                       epochs=50,
                                       batch_size=10,
                                       train_whole_model=True)

        # evaluate
        model.evaluate(train_data)
        model.evaluate(valid_data)

        # export the model
        model.export(export_dir='.',
                     tflite_filename='efficientdet-lite-bfc.tflite',
                     label_filename='bfc-labels.txt',
                     export_format=[ExportFormat.TFLITE, ExportFormat.LABEL])

        # evaluate model using tflite
        model.evaluate_tflite('efficientdet-lite-bfc.tflite', train_data)
        model.evaluate_tflite('efficientdet-lite-bfc.tflite', valid_data)



