FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel
MAINTAINER Viet Nguyen <nhviet1009@gmail.com>

RUN apt-get update -y && apt-get install -y libglib2.0-dev libsm6 libxext6 libxrender-dev freeglut3-dev ffmpeg
RUN pip install gymnasium==1.3.0 gym-super-mario-bros==9.1.0 nes-py==9.0.1 opencv-python==4.14.0.94 future==0.18.2 pyglet==1.5.7

WORKDIR /Super-mario-bros-PPO-pytorch
