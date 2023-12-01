# Deep_Orga
Deep_Orga model maintains the single-stage and concise features of the YOLOX model while surpassing the performance of the classical models.

![Total pipline of the Deep_Orga for organoid detection](/Graphical Abstract.tif)

### Screenshot of the User Interface
![Screenshot of User Interface](/readme_images/screenshot.jpg)

## Getting Started
### Hardware Prerequisites
We recommend a computer with at least 16 GB of RAM and an NVIDIA graphics card (required) with at least 8 GB of GPU memory.
All code was developed and tested in a Windows environment, but should work fine on both Mac OS and Linux.

### Installation
1. Install Anaconda from https://www.anaconda.com/distribution/
2. Open the Anaconda Prompt and create a new conda environment using `conda create -n orgaquant python=3.6`
3. Activate the newly created environment using `activate orgaquant`
4. Install Tensorflow and Git using `conda install tensorflow-gpu=1.14 git`
5. Install dependencies using `pip install keras-resnet==0.2.0 cython keras matplotlib opencv-python progressbar2 streamlit`
6. Clone the OrgaQuant repository using `git clone https://github.com/TKassis/OrgaQuant.git`
7. Move into the directory using `cd OrgaQuant`
8. Install _keras_retinanet_ using `python setup.py build_ext --inplace`. More information [here](https://github.com/fizyr/keras-retinanet)
9. Download the pre-trained model from [here](https://github.com/TKassis/OrgaQuant/releases/download/v0.2/orgaquant_intestinal_v3.h5) and place inside the _trained_models_ folder.
