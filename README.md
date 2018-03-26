# Style-Transfer
This is an implementation of the paper "Image Style Transfer Using Convolutional Neural Networks", authored by Gatys et al. This project was done using Tensorflow. A pretrained VGG16 model was used. The max pool layers were replaced by average pooling, and the Adam Optimizer was used instead of L-BFGS which was used in the paper.


## Files

### model.py
This contains the code for the pretrained VGG16 model. The weights for the model were downloaded from [here](https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM), and the model was subsequently built using the aforementioned script. The fully connected layers have been commented out, because they are not used in the application, and would lead to errors if images of size other than (224,224,3) are used.

### style_transfer.py
The training of the network is defined here. The paper describes two losses - the content loss and style loss. A weight is associated with each of these classes to denote the ratio of content and style that is desired in the final image. Starting from a noisy input, the network is trained to achieve a visually appealing image that describes the content in the given style.

### test.py
This is the main function, containing the various hyperparameters and initialisations. Running this script will begin the training.
