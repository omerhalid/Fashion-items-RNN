Fashion Item Recognition with Vanilla RNN

This is a simple implementation of a vanilla RNN for recognizing fashion items using the Fashion-MNIST dataset. The RNN is trained to predict the category of the item (one of 10 categories) based on its image. This project is implemented in Python using the PyTorch library.
Getting Started

To get started with this project, you'll need to do the following:

    Clone this repository to your local machine.
    Install the required packages listed in requirements.txt by running the following command:

pip install -r requirements.txt

    Run the main.py script to train the RNN and test its performance on the Fashion-MNIST dataset.

Files

This repository contains the following files:

    main.py: This is the main script that trains the RNN and tests its performance on the Fashion-MNIST dataset.
    model.py: This file defines the architecture of the RNN model.
    utils.py: This file contains utility functions for loading the Fashion-MNIST dataset and processing the data.
    requirements.txt: This file lists the required packages and their versions.

Hyperparameters

The following hyperparameters can be adjusted in the main.py script:

    input_size: The size of the input vector. This is set to 28, as each Fashion-MNIST image has 28x28 pixels.
    sequence_length: The length of the input sequence. This is set to 28, as each image is flattened into a sequence of 28 vectors.
    hidden_size: The number of hidden units in the RNN. This is set to 128.
    num_layers: The number of layers in the RNN. This is set to 2.
    num_classes: The number of categories in the Fashion-MNIST dataset. This is set to 10.
    batch_size: The size of the mini-batch used for training. This is set to 100.
    num_epochs: The number of epochs to train the RNN. This is set to 10.
    learning_rate: The learning rate for the optimizer. This is set to 0.001.

Results

After training the RNN for 10 epochs, the model achieves an accuracy of approximately 84% on the test set.
Acknowledgements

This project is based on the Fashion-MNIST dataset, which can be found at https://github.com/zalandoresearch/fashion-mnist.
License

This project is licensed under the MIT License. See the LICENSE file for details.
