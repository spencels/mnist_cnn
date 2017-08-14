# MNIST Classifier

Convolutional neural network implemented in tensorflow to classify handwritten
digits from the MNIST dataset. Implements several techniques common in
state-of-the-art document classifiers:

* Convolutional neural network [(Ciresan et al. 2012)](https://arxiv.org/pdf/1202.2745.pdf)
* Elastic distortion data augmentation [(Simard et al., ICDAR 2003)](https://pdfs.semanticscholar.org/7b1c/c19dec9289c66e7ab45e80e8c42273509ab6.pdf)
* Multi-column neural network [(Ciresan et al. 2012)](https://arxiv.org/pdf/1202.2745.pdf), without the width normalization
* Dropout regularization [(Tensorflow tutorial)](https://www.tensorflow.org/get_started/mnist/pros)

The best result achieved was a 0.39% error, though due to a bug in this version
of the code training error to be 1%.

## Running

Requirements:

* Python 3
* Tensorflow

Create a virtual environment

```sh
$ virtualenv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

Download [MNIST data files](http://yann.lecun.com/exdb/mnist/}) to a folder
named `MNIST_data`. Run the preprocessing script:

```sh
$ python process_data.py
```

Train and evaluate:

```sh
$ python mnist_basic.py --train
```