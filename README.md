[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-360/)    

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)


# Application of RL Double Q-Network on Flappy Bird.
This assignment aims at exploring the basic principles of Reinforcement Learning and specifically DDQN method. For this purpose an agent is trained on the Flappy Bird game, which consists of two actions: fly and do nothing. A convolutional based model architecture is utilized and trained. This repo provides the tools and not a perfect final result. One can explore the process of training (`steps.odt`) and try experimenting further. A suggestion would be to make the training more stable by adding deeper or more layers. The basic parameters for getting started can be found on `tests.csv` test 22.

https://github.com/Xritsos/Flappy-RL/assets/57326163/9326ae1d-a5a2-41aa-95fe-4df22494e45e

# Files
All of the game assets such as sounds and images can be found on `assets`, while logs are kept for each test tried on `tests.csv`, on the `logs` folder. The best model so far is saved in the `model_ckpts` folder, although an investigation should be made, because there seems that model saving is problematic. To test and visualize the results of your model access `test/test_game.py`, where the model is loaded and an avi file is saved for each trial.  

Regarding the game, the pygame library is used as a base and the Flappy Bird game can be found on `source/game`. To train your own model, access `source/DoubleDQN.py` and pass a test id that is read from the `tests.csv` file. In order to customize your own model architecture, access `source/models/dqnet.py`. Image preprocessing, replay buffer memory and plotting functions can be found on `source/utils`.  

# Requirements
The whole training process was held on linux machine. GPU utilization is advised as the learning process can be very time consuming. You can use the `requirements.txt` file to setup your environment and in case you face any problem visualizing the pygame screen, visit the following thread: https://stackoverflow.com/questions/72110384/libgl-error-mesa-loader-failed-to-open-iris.

## Acknowledgements
Results presented in this work have been produced using the AUTH Compute Infrastructure and Resources. The author would like to acknowledge the support provided by the Scientific Computing Office throughout the progress of this research work.

## Refences
[1] https://github.com/uvipen/Flappy-bird-deep-Q-learning-pytorch/tree/master  
[2] https://github.com/hardlyrichie/pytorch-flappy-bird/tree/master  
[3] https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html  
[4] https://www.lesswrong.com/posts/kyvCNgx9oAwJCuevo/deep-q-networks-explained

## Contact
e-mail: chrispsyc@yahoo.com
