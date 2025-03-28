# Python Neuralink
This repository contains the Python Neuralink Library and samples.
## How to work with the library?
You need to clone this repository into your project. then import NeuralNetwork class from neural_network.py.
### How does NeuralNetwork works?
NeuralNetwork inits Layers and connects them during forward and back propogation. It happens automatically, there is no need to know it for simple users.
Layers (see folder layers) connect operations, calculate gradients for them.In actual version of projects there are 2 types of layers Layer (simple class, requires to be inheritant) and Dense (Layer with WeightsMultiply, BiasAddand activation (or without) operations). 
Operations (see folder operations) calculate forward and backward values. There are 5 types of operations at the moment:
<1. Operation. Simple operation class, requres to be inheritant.
<2. ParamOperation. Inherited from Operation. Creates ability to work with operations with parametres. Also contatins function to calculate gradients for them.
<3. WeightsMultiply. Inhereted from ParamOperation. Creates weights matrices and multiply input matrice on them.
<4. BiasAdd. Inhereted from ParamOperation. Creates Bias matrice ans sum it with input matrice. Usually works with WeightsMultiply.
<5. Sigmoid. Inhereted from Operation. Simply modifies input matrice accoridng as a Sigmoid function. Should be Activation in Dense. 
### How to train it?
To train NeuralNetwork see trainer.py. The class Trainer generate batches and trains NeuralNetwork, till the loss increases. You can modify the training process experemnting with parametres.
## How to work with samples?
Firstly you should run train.py to train neuralink. Then you can open test.py to check the neuralink abilities with your own cases.  
