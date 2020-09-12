# tensorflow-cxx-mnist

A TensorFlow 2 plain C++ example for training and classifying MNIST data. Uses the Adam optimizer.

Uploaded as I did not find many examples on how to use TensorFlow in plain C++ for building and training models.

When adapting this example for Linux, the first lines of the C++ code have to be removed as they are Windows specific.
Instead you will have to add the correct Linux headers (maybe only arpa/inet.h for ntohl).
