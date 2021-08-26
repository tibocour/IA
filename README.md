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

Les jeux de données sont au format [Pascal VOC XML](http://host.robots.ox.ac.uk/pascal/VOC/). 
Ce format peut typiquement etre généré par l'outil [Label Studio](https://labelstud.io/)

## Demonstrateur 

![Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white)

Un démonstrateur utilisant [`detectron2`](https://github.com/facebookresearch/detectron2) construit sur 
[`PyTorch`](https://github.com/pytorch/pytorch) est disponible dans un
notebook mis au point sur [`Google/Colab`](https://colab.research.google.com/). 
Voir [ici](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb#scrollTo=Rmai0dD30XzL)
pour les détails d'intégration.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tibocour/IA/blob/master/notebooks/detectron2.ipynb)
[Demonstrateur - apprentissage par detectron2 pour la détection de mégots](https://github.com/tibocour/IA/blob/master/notebooks/detectron2.ipynb)

## Entrainement

### Demonstrateur

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

## Inférence Coral TPU

Coming soon !! 

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

## Contacts

email : [birdsforchange](mailto:contact@birdsforchange.com)

auteurs : [@tibocour](https://github.com/tibocour), [@sdesrozis](https://github.com/sdesrozis)


