# Advanced-Classification

 Yong Kai Dong
 
 Purdue Email
 
 https://github.com/YongDong11467/Advanced-Classification
 
Resources: 
  - colab.research.google.com
  - https://www.tensorflow.org/api_docs/python/tf/keras
  - https://www.youtube.com/watch?v=iedmZlFxjfA&list=WL&index=230&t=180s

Completed:
  - Neural net models
  - Accuracy
      - mnist_d (99.05%)
      - mnist_f (93.12%)
      - cifar_10 (76.69%)
      - cifar_100_f (41.12%)
      - cifar_100_c (50.88%)
  - Pipeline & misc (No Ann plot)
  - Report

How is a CNN superior to standard ANNs for image processing?
  - CNN has an additional layer called the conv layer which does additional computation by running the images through filters.
  
Why do we sometimes use pooling in CNNs?
  - We sometimes use pooling to reduce the size of the data.
  
Why do you think the cifar datasets are harder than mnist?
  - Cifar datasets contains an extra dimension that is used to represent the different colors channels.
  
Explain how you increases the accuracy of your CNN
  - The accuracy of CNN can be increased by increasing the hyperparameters, adding/removing layers, and increasing the training time. When my accuracy gets extremely close to the
    target I would first try increasing the eps. If it goes down then it means I'm overtraining. The next solution would just to change around the other hyperparameters until
    the target percentage is reached.
    
Hyperparameters
  - Filter size
  - Stride
  - Padding
  - Dropout Rate
  - Number of epochs
