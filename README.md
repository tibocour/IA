# Birds For Change 

## Détection de déchets par Deep Learning

Ce projet regroupe les développements effectués par [`IFP Energies nouvelles (IFPEN)`](https://www.ifpenergiesnouvelles.fr/) pour la détection
de déchets dans des images typiquement issues d'automates mis en place par 
[`Birds For Change (BFC)`](https://www.birdsforchange.fr/).

![BFC](assets/bfc-logo.png)
![IFPEN](assets/ifpen-logo.jpg)

## Jeu de données

![Example de mégot](assets/megot-sample.png)

Ensemble de jeux de données annotées par `BFC` disponibles :
* [`megots150images.zip`](data/megots150images.zip) : archive de 147 images de mégots de cigarette
* [`train_megots150images.zip`](data/train_megots150images.zip) : archive de plus de mille images de mégots de cigarette générée à partir de `megots150images.zip` pour l'entrainement
* [`valid_megots150images.zip`](data/valid_megots150images.zip) : archive de 17 images de mégots de cigarette générée à partir de `megots150images.zip` pour la validation

Les jeux de données sont au format [Pascal VOC XML](http://host.robots.ox.ac.uk/pascal/VOC/). 
Ce format peut typiquement etre généré par l'outil [Label Studio](https://labelstud.io/)

Pour être valide, une archive `zip` doit contenir 2 répertoires `images` et `Annotations`.

```commandline
> unzip megots150images.zip

Archive:  megots150images.zip
   creating: Annotations/
   creating: images/
  inflating: images/1608647944.jpg   
  ...
  
> tree
.
├── Annotations
│   ├── 1608644703.xml
│   ├── ...
├── images
│   ├── 1608644703.jpg
│   ├── ...
└── megots150images.zip
```

## Demonstrateurs

## `detectron2`

![Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white)

Un démonstrateur utilisant [`detectron2`](https://github.com/facebookresearch/detectron2) construit sur 
[`PyTorch`](https://github.com/pytorch/pytorch) est disponible dans un
notebook mis au point sur [`Google/Colab`](https://colab.research.google.com/). 
Voir [ici](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb#scrollTo=Rmai0dD30XzL)
pour les détails d'intégration.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tibocour/IA/blob/master/notebooks/detectron2.ipynb)
[Demonstrateur - apprentissage par detectron2 pour la détection de mégots](https://github.com/tibocour/IA/blob/master/notebooks/detectron2.ipynb)

## `tflite-model-maker`

![Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)

Un entrainement utilisant [`tflite-model-maker`](https://www.tensorflow.org/lite/tutorials/model_maker_object_detection) construit sur 
[`Tensorflow`](https://www.tensorflow.org) est disponible dans un
notebook mis au point sur [`Google/Colab`](https://colab.research.google.com/). 
Voir [ici](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb#scrollTo=Rmai0dD30XzL)
pour les détails d'intégration.

> La version de `tensorflow` utilisée est [`tensorflow-lite`](https://www.tensorflow.org/lite) qui est une version
> lègère spécifiquement mis au point pour les machines mobiles, et IoT. Ceci limite 
> en particulier le choix des architectures neuronales.

On se base ici sur l'architecture de detection proposée par `Google` : [EfficientDet](https://arxiv.org/pdf/1911.09070.pdf). 
Voir également [ici](https://ai.googleblog.com/2020/04/efficientdet-towards-scalable-and.html) pour les détails.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tibocour/IA/blob/master/notebooks/tflite_model_maker.ipynb)
[Apprentissage par tensorflow-lite pour la détection de mégots](https://github.com/tibocour/IA/blob/master/notebooks/tflite_model_maker.ipynb)

## Entrainement

## Installation

Les modules nécessaires s'installent via `pip`

    pip install -r requirements.txt

## Etape 0 - Augmentation des données

Si la librairie `detectron2` intègre automatiquement l'augmentation de données, ce n'est pas (encore) le cas de 
`tensorflow-lite`. Pour etre exact, cette fonctionnalité est disponible pour la classification mais pas encore pour
la détection. 

D'autre part, le format `Label Studio` est incompatible pour une utilisation directe dans `tflite-model-maker` à cause :
* utilisation du format d'encoding `xml` incompatible avec la librairie de parsing `lxml`
* les fichiers n'ont pas l'extension `.jpg`

Le script `python/label_studio_voc_converter.py` permet de corriger cela et d'augmenter les données en une passe. Les 
augmentations sont typiquement des flips, crops et autres modifications affines. De plus, le script permet de séparer 
les données pour l'entrainement est la validation suivant un ratio.

Usage :

    python python/label_studio_voc_converter.py --zip <label-studio-zip-file>
                                                --train_split <ratio>
                                                --size <nb-of-augmentation>

Défauts :
* `--train_split 0.8` : 80% des données sont pour l'apprentissage.
* `--size 10` : 10 images augmentées sont générées en plus de l'originale.

> Les augmentations ne sont pas opérées sur les données de validation.

Exemple :

    python python/label_studio_voc_converter.py --zip data/megots150images.zip
    
Les fichiers `train_megots150images.zip` et `valid_megots150images.zip` sont alors générés.
Chacun est un dataset au format `Pascal VOC XML`.

Un notebook a été mis au point pour tester le script d'augmentation.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tibocour/IA/blob/master/notebooks/data_augmentation.ipynb)
[Augmentationn de données](https://github.com/tibocour/IA/blob/master/notebooks/data_augmentation.ipynb)

## Etape 1 - Entraintement par Transfert Learning

Le script `python/train.py` permet d'effectuer un entrainement d'un réseau de neurones pour la détection. Le 
réseau utilisé est `efficientdet`, préconisé par Google pour la détection sur Coral. Pour plus de détails,
voir la [publication](https://arxiv.org/pdf/1911.09070.pdf) et 
[ici](https://ai.googleblog.com/2020/04/efficientdet-towards-scalable-and.html).

Usage :

    python python/train.py --train_zip <train-zip-path>
                           --valid_zip <valid-zip-path> 
                           --epochs <nb-of-epochs>
                           --batch_size <batch-size>
                           --with_tflite_eval <0-or-1>

Défauts :
* `--epochs 50` : apprentissage sur 50 epochs
* `--batch_size 10` : Batch de 10 images
* `--with_tflite_eval 0` : pas d'évaluation du modèle `tflite` par défaut (couteux). 

Exemple :
    
    python python/train.py --train_zip data/train_megots150images.zip
                           --valid_zip data/valid_megots150images.zip

Le script permet de générer un fichier du modèle `efficientdet-lite-bfc.tflite` et un fichier `bfc-labels.txt` 
contenant les labels. En particulier, c'est le fichier `tflite` qui doit etre utilisé en inférence.

> Pour execution sur `Coral TPU`, le modèle compilé doit maintenant être compilé. 
> Voir la section [Compilation du modèle](#compilation-du-modèle)

## Etape 2a - Inférence d'une image

Le script `python/google-coral-inference/detect_image.py` permet de lancer une inférence d'une image. Ce script est un 
fork de la version officielle proposée par `google-coral`. Toutefois, une adaptation permet de lancer des inférences 
hors `Coral TPU`. 

Usage :

    cd IA/python/google-coral-inference/

    python detect_image.py --input <image-path>
                           --output <output-path>
                           --model <tflite-model-path>
                           --labels <labels-path>
                                   
> A noter qu'il faut lancer l'inférence à partir du dossier du script

Exemple :

    cd IA/python/google-coral-inference/

    python detect_image.py --input ../../data/images/test_image.jpg
                           --model ../../data/efficientdet-lite-bfc.tflite
                           --labels ../../data/bfc-labels.txt
                           --output ./inference_test_image.jpg

> Testé sur x86 et Google Colab.

> Pour execution sur `Coral TPU`, utiliser le modèle compilé `efficientdet-lite-bfc_edgetpu.tflite`.

## Etape 2b - Inférence d'une vidéo

Le script `python/google-coral-inference/detect_video.py` permet de lancer une inférence d'une vidéo. Ce script est une
modification de la version officielle proposée par `google-coral`. 

Usage :

    cd IA/python/google-coral-inference/

    python detect_video.py --input <image-path> | --camera <camera-id>
                           --output <output-path>
                           --model <tflite-model-path>
                           --labels <labels-path>
                                   
> A noter qu'il faut lancer l'inférence à partir du dossier du script

Exemple :

    cd IA/python/google-coral-inference/

    python detect_video.py --input ../../data/1megot.mp4
                           --model ../../data/efficientdet-lite-bfc.tflite
                           --labels ../../data/bfc-labels.txt
                           --output ./inference_1megot.mp4

> Testé sur x86 et Google Colab.

> Pour execution sur `Coral TPU`, utiliser le modèle compilé `efficientdet-lite-bfc_edgetpu.tflite`.

Pour tester à partir d'une caméra donnée :

    cd IA/python/google-coral-inference/

    python detect_video.py --camera 0
                           --model ../../data/efficientdet-lite-bfc.tflite
                           --labels ../../data/bfc-labels.txt
                           --output ./inference_camera.mp4

## Notebooks des scripts

Un notebook de synthèse a été mis au point pour tester les scripts d'entrainement (Etape 1) et d'inférence d'une 
image (Etape 2a).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tibocour/IA/blob/master/notebooks/train_tflite_model.ipynb)
[Apprentissage et inférence d'une image par scripts](https://github.com/tibocour/IA/blob/master/notebooks/train_tflite_model.ipynb)

Un notebook de synthèse a été mis au point pour tester les scripts d'inférence d'une image (Etape 2a) et vidéo
(Etape 2b).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tibocour/IA/blob/master/notebooks/infer_tflite_model_video.ipynb)
[inférence d'une image et une vidéo par scripts](https://github.com/tibocour/IA/blob/master/notebooks/infer_tflite_model_video.ipynb)

## Inférence Coral TPU

Pour utiliser l'inférence sur `Coral TPU`, il faut au préalable installer les différents modules nécessaires en suivant
les [consignes suivantes](https://github.com/google-coral/tflite/tree/master/python/examples/detection#set-up-your-device).

En particulier, voir 
* https://coral.ai/docs/accelerator/get-started/
* https://www.tensorflow.org/lite/guide/python

L'inférence présentée dans la section précédente fonctionne en tirant partie du `Coral TPU`.

### Compilation du modèle

Les modèles appris dans les sections précédentes doivent maintenant être compilés spécifiquement pour tirer parti du
`Coral TPU`. On peut installer le compilateur `Edge TPU` en suivant les [consignes suivantes](ttps://coral.ai/docs/edgetpu/compiler/#system-requirements).

> Attention, le compilateur n'est disponible que sur les systèmes basés sur Linux/Debian. Une alternative consiste à 
> passer par Google Colab pour effectuer la compilation. Pour cela, ouvrir et exécuter le notebook 
> [ici](https://colab.research.google.com/github/google-coral/tutorials/blob/master/compile_for_edgetpu.ipynb). 

Typiquement, les modèles compilés sont suffixés par `_edgetpu.tflite`.

## Ressources en vrac

* https://colab.research.google.com/
* http://host.robots.ox.ac.uk/pascal/VOC/
* https://labelstud.io/
* https://pytorch.org/docs/stable/index.html
* https://github.com/pytorch/pytorch
* https://pytorch.org/tutorials/
* https://pytorch.org/ecosystem/
* https://github.com/facebookresearch/detectron2
* https://ai.facebook.com/blog/-detectron2-a-pytorch-based-modular-object-detection-library-/
* https://detectron2.readthedocs.io/en/latest/index.html
* https://www.tensorflow.org/lite/tutorials/model_maker_object_detection
* https://www.tensorflow.org/lite/api_docs/python/tflite_model_maker/object_detector
* https://coral.ai/docs/edgetpu/retrain-detection/
* https://github.com/google-coral/tflite/tree/master/python/examples/detection
* https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/raspberry_pi

Lien à noter pour l'apprentissage à partir de `tensorflow` (conversion vers `tflite` ensuite)
* https://github.com/tensorflow/models/tree/master/research/object_detection/models
* https://github.com/tensorflow/models/tree/master/research/object_detection/colab_tutorials

> Tutoriel `eager_few_shot_od_training_tflite.ipynb` à voir si besoin

Consigne d'installation pour `Coral TPU`
* https://coral.ai/docs/accelerator/get-started/
* https://www.tensorflow.org/lite/guide/python
* https://github.com/google-coral/tflite/tree/master/python/examples/detection

Exemples d'application `Coral TPU`
* https://github.com/google-coral/pycoral
* https://github.com/google-coral/examples-camera

## Contacts

email : [birdsforchange](mailto:contact@birdsforchange.com)

auteurs : [@tibocour](https://github.com/tibocour), [@sdesrozis](https://github.com/sdesrozis)


