# 3D Reconstruction
This project aims to reconstruct a 3D scene by finding corresponding points in consecutive 2D images. The points of the scene are visualized and the camera poses are evaluated against the ground truth provided for the scene.

Correspondences between images are found using *Harris Interest Points* and used to estimate the fundamental matrix *F* by running the *Gold Standard Algorithm*. New views are added using the *PnP Algorithm* and refined by *Bundle Adjustment*.

![image](https://github.com/user-attachments/assets/6bc69f1e-2afa-40f5-941c-8e46fb336355)




## To build with Ceres and Pybind:

### Mac
- Install with homebrew:
    - brew install ceres-solver
    - brew install cmake
    - brew install glog
    - brew install eigen
    - brew install suite-sparse
- mkdir build
- cd build
- cmake ..
- make
- run main.py

### Linux (Not tested yet)
- Install:
    - sudo apt-get install cmake
    - sudo apt-get install libgoogle-glog-dev
    - sudo apt-get install libatlas-base-dev
    - sudo apt-get install libeigen3-dev
    - sudo apt-get install libsuitesparse-dev
- mkdir build
- cd build
- cmake ..
- make
- run main.py

