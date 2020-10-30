__author__ = 'Corey Schaf <coreyjs@hey.com>'
__version__ = '1.0.0'
__license__ = 'MIT'

# Train a new network on a data set with train.py
#
#     Basic usage: python train.py data_directory
#     Prints out training loss, validation loss, and validation accuracy as the network trains
#     Options:
#         Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
#         Choose architecture: python train.py data_dir --arch "vgg13"
#         Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
#         Use GPU for training: python train.py data_dir --gpu


import os
import argparse
import network_helper as nh

# Build up our available arguments for our training CLI tool.
parser = argparse.ArgumentParser(description='Train a network on a dataset',
                                 usage='python train.py ./data_sets/train --gpu --learning_rate 0.02 '
                                       '--epohcs 10 --arch vgg11')
parser.add_argument('data_dir', type=str, help='Directory of the training images')
parser.add_argument('--gpu', action='store_true', default=False, help='Use GPU for training, defaults to False')
parser.add_argument('--arch', type=str, default='vgg19',
                    help='Choose an architecture for the network, defaults to VGG19')
parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Set directory to save checkpoints')
parser.add_argument('--learning_rate', type=float, default=0.01,
                    help='Set the learning rate hyperparameter, defaults to 0.01')
parser.add_argument('--hidden_units', type=int, default=512, help='Set the hidden unit amount, defaults to 512')
parser.add_argument('--epochs', type=int, default=20,
                    help='Set the total amount of epochs this network should train for, defaults to 20')

args = parser.parse_args()


def main(args):
    # Validate that our supplied architecture is supported
    if not nh.validate_arch(arch=args.arch):
        print(f'Invalid architecture type supplied, must be of type: {nh.arch_types}')

    use_gpu: bool = args.gpu
    print('Info -- Using GPU') if use_gpu else print('Info -- Using CPU')
    print(f'Info -- ({args.epochs}) Epochs.  Learning Rate: {args.learning_rate}.  '
          f'Hidden Units: {args.hidden_units}.  NN Arch: {args.arch}')

    # Check for checkpoint directory, create if it DNE
    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    # Load the dataset.  This will be assuming a directory with sub directories of
    # images per classification.  i.e. /data/images/train/3/fooflower.jpg, where 3 is classification.
    image_datasets, dataloaders = nh.load_dataset(args.data_dir)
    class_count = len(image_datasets['train'].classes)
    model = nh.create_model(arch_type=args.arch, hidden_units=args.hidden_units, out_features=class_count)
    model, optimizer = nh.train_model(model=model, use_gpu=use_gpu, epochs=args.epochs, dataloader=dataloaders)

    nh.save_checkpoint(model=model, path=args.save_dir, image_datasets=image_datasets,
                       epochs=args.epochs, optimizer=optimizer, arch=args.arch)


if __name__ == '__main__':
    main(args)
