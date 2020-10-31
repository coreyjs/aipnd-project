# AI Programming - Image Classification Neural Network


> Project code for Udacity's AI Programming with Python Nanodegree program.

![Screenshot 2020-10-31 101251](https://user-images.githubusercontent.com/1228838/97781440-c40e9600-1b61-11eb-9bc5-4096fe527443.png)
![Screenshot 2020-10-31 1013243](https://user-images.githubusercontent.com/1228838/97781441-c40e9600-1b61-11eb-8bc7-272af114d698.png)


In this project we train an image classifier to recognize the different species of flowers.
The first part of this project can be found in the notebook `Image Classifier Project.ipynb`.  Here is
where we work through defining the neural network architecture, process our images and
train/validate and test our network's accuracy.

> Note, an HTML version of this notebook can be found here: 'Image Classifier Project.html'

---
Along with the notebook this classification network can be ran on the command line. 

For the dataset we are using, you can set those up with the following:

```bash
mkdir -p data_images && cd data_images
curl https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz | tar xz
```

### 1. Train

Train a new network on a data set with train.py

 - Basic usage: python train.py data_directory
 - Prints out training loss, validation loss, and validation accuracy as the network trains
    Options:
       
  Set directory to save checkpoints: `python train.py data_dir --save_dir save_directory`
   
  Choose architecture: `python train.py data_dir --arch "vgg13"`
   
  Set hyperparameters: `python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20`
   
  Use GPU for training: `python train.py data_dir --gpu`
 
```bash
usage: python train.py ./data_sets/train --gpu --learning_rate 0.02 --epohcs 10 --arch vgg11

Train a network on a dataset

positional arguments:
  data_dir              Directory of the training images

optional arguments:
  -h, --help            show this help message and exit
  --gpu                 Use GPU for training, defaults to False
  --arch ARCH           Choose an architecture for the network, defaults to VGG19
  --save_dir SAVE_DIR   Set directory to save checkpoints
  --learning_rate LEARNING_RATE
                        Set the learning rate hyperparameter, defaults to 0.01
  --hidden_units HIDDEN_UNITS
                        Set the hidden unit amount, defaults to 512
  --epochs EPOCHS       Set the total amount of epochs this network should train for, defaults to 20
```
---
Available Architecture's:
```python
arch_types = {
    'densenet121': models.densenet121,
    'densenet169': models.densenet169,
    'densenet161': models.densenet161,
    'densenet201': models.densenet201,
    'vgg11': models.vgg11,
    'vgg13': models.vgg13,
    'vgg16': models.vgg16,
    'vgg19': models.vgg19,
}
```
---

Example, training the network:

```bash
> python train.py ./data_images --arch "vgg19" --learning_rate 0.01 --gpu --epochs 10
---
Info -- Using GPU
Info -- (13) Epochs.  Learning Rate: 0.01.  Hidden Units: 2048.  NN Arch: vgg19
1/13 Epochs
Epoch: 1/13  Training Loss: 6.293  Validation Loss: 4.084  Validation Accuracy: 0.166
Epoch: 1/13  Training Loss: 4.017  Validation Loss: 3.147  Validation Accuracy: 0.290
```

### Checkpoints

When training a neural network it will automatically save a checkpoint file in `./checkpoints/checkpoint-[ARCHNAME].pth`.
You can override this with the flag `--save_dir /my/save/dir`.

---

### 2. Prediction
Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

 Basic usage: python predict.py /path/to/image checkpoint
 Options:
   - Return top KKK most likely classes: `python predict.py input checkpoint --top_k 3`
   - Use a mapping of categories to real names: `python predict.py input checkpoint --category_names cat_to_name.json`
   - Use GPU for inference: `python predict.py input checkpoint --gpu`
   
```
usage: python predict.py ./data_sets/valid/1/img.jpg

Predict a flower name from an image along with the probability of that name.

positional arguments:
  image_path            Path to image to predict
  checkpoint            Checkpoint file location

optional arguments:
  -h, --help            show this help message and exit
  --gpu                 Use GPU for training, defaults to False
  --top_k TOP_K         Return top K most likely classes
  --category_names CATEGORY_NAMES
                        Mapping file of class id to real names, defaults to cat_to_name.json
```

Example of predicting with a supplied image:

```bash
>  python predict.py ./data_images/valid/71/image_04517.jpg ./checkpoints/checkpoint-vgg19.pth --gpu
--
Info -- Using GPU
Image resized to: (256, 191), from: (667, 500)
left: 16.0, upper: -16.5, right: 240.0, lower: 207.5
---------------Results-----------------
Prediction: GAZANIA(71) with accuracy 0.99999
Prediction: BLACK-EYED SUSAN(63) with accuracy 0.00001
Prediction: ENGLISH MARIGOLD(5) with accuracy 0.00000
---------------------------------------
```

