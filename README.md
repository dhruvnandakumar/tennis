# tennis
An MADDPG Implementation to solve Unity's Tennis Environment

## 1 Project Details 
This project aims to sold Unity's Tennis environment which consists of two tennis rackets playing tennist. The agents receive +0.1 points for every time an agent hits the ball above the net and -0.01 for every time the ball hits the net or goes out of points. The aboservation space consist of 8 variables consistsing of agent and ball velocity and position.

The game is considered solved with the average score (of the highest score acheived by each agent) over 100 episosed (games) is over +0.5. 

## 2 Getting started with the repository
This repository requires some dependencies to be installed, particulary the Unity Gym. 
Follow these links to install the right dependencies: 
1. Linux: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip
2. Mac OSX: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip
3. Windows 32bit: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip
4. Windows 64bit: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip
5. No monitor linux: https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip

You will also have to install some python depencies:
1. First, please follow the setup on this url to get your python virtual environment set up: https://github.com/udacity/deep-reinforcement-learning#dependencies
2. Open a terminal window at the root of this repository and run the following command: pip -q install ./python

## 3 Instructions 

Place the download in this repository in the root folder, and follow along with the tennis.ipynb Jupyter Notebook. 
