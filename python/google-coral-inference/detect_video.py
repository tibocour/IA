# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example using TF Lite to detect objects in a given image."""

import argparse
import time

import cv2

from PIL import Image
from PIL import ImageDraw

import numpy as np

import detect

try:
    import tflite_runtime.interpreter as tflite
    import platform

    EDGETPU_SHARED_LIB = {
        'Linux': 'libedgetpu.so.1',
        'Darwin': 'libedgetpu.1.dylib',
        'Windows': 'edgetpu.dll'
    }[platform.system()]

    with_edgetpu = True
except ImportError:
    import tensorflow.lite as tflite

    with_edgetpu = False


def load_labels(path, encoding='utf-8'):
    """Loads labels from file (with or without index numbers).

    Args:
      path: path to label file.
      encoding: label file encoding.
    Returns:
      Dictionary mapping indices to labels.
    """
    with open(path, 'r', encoding=encoding) as f:
        lines = f.readlines()
        if not lines:
            return {}

        if lines[0].split(' ', maxsplit=1)[0].isdigit():
            pairs = [line.split(' ', maxsplit=1) for line in lines]
            return {int(index): label.strip() for index, label in pairs}
        else:
            return {index: line.strip() for index, line in enumerate(lines)}


def make_interpreter(model_file):
    model_file, *device = model_file.split('@')
    kwargs = {
        "model_path": model_file,
    }
    if with_edgetpu:
        kwargs.update({
            "experimental_delegates": [
                tflite.load_delegate(EDGETPU_SHARED_LIB, {'device': device[0]} if device else {})
            ]
        })
    return tflite.Interpreter(**kwargs)


def draw_objects(draw, objs, labels):
    """Draws the bounding box and label for each object."""
    for obj in objs:
        bbox = obj.bbox
        draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                       outline='red')
        draw.text((bbox.xmin + 10, bbox.ymin + 10),
                  '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score),
                  fill='red')


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', required=True,
                        help='File path of .tflite file.')
    parser.add_argument('-i', '--input', required=True,
                        help='File path of video to process.')
    parser.add_argument('-l', '--labels',
                        help='File path of labels file.')
    parser.add_argument('-t', '--threshold', type=float, default=0.8,
                        help='Score threshold for detected objects.')
    parser.add_argument('-o', '--output',
                        help='File path for the result video with annotations')
    args = parser.parse_args()

    labels = load_labels(args.labels) if args.labels else {}
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()

    cap = cv2.VideoCapture(args.input)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_writer = cv2.VideoWriter(args.output,
                                   cv2.VideoWriter_fourcc(*'MP4V'),
                                   fps, (w, h))

    frame_id = 0

    while True:

        is_ok, img0 = cap.read()

        if not is_ok:
            break

        img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(img)
        scale = detect.set_input(interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))

        start = time.perf_counter()
        interpreter.invoke()
        inference_time = time.perf_counter() - start
        print(f"inference time of the frame {frame_id}: {inference_time * 1000:2f} ms")

        objs = detect.get_output(interpreter, 0.8, scale)

        if not objs:
            print(f"\tNo objects detected in the frame {frame_id}")
        else:
            for obj in objs:
                print(f"\t{len(objs)} object(s) detected in the frame {frame_id}")
                print(f"\t  label: {labels.get(obj.id, obj.id)}")
                print(f"\t  id:    {obj.id}")
                print(f"\t  score: {obj.score}")
                print(f"\t  bbox:  {obj.bbox}")
            print("add bbox to the frame {frame_id}")
            draw_objects(ImageDraw.Draw(image), objs, labels)
            img0 = np.asarray(image)
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

        print(f"write the frame {frame_id}\n")
        video_writer.write(img0)

        frame_id += 1

    print("release the video")
    video_writer.release()


if __name__ == '__main__':
    main()
