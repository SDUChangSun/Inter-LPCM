# Inter-LPCM
This repository contains the official research code for a learning-based point cloud compression framework. The proposed system integrates point cloud registration, deep neural networks, and entropy coding to efficiently compress 3D point clouds. The code is intended for testing and experimental evaluation purposes only.

Repository Structure

├── model.py # predictive model for theta
├── r_model.py # predictive model for r
├── r_entropy_model.py # entropy model for r
├── phi_entropy_model.py # entropy model for phi
├── s_r2.py # point cloud registration module
├── get_t.py # get transformation matrix t
├── nearpoint.py # nearest-point processing utilities
├── dataprepare/
│ └── data.py # dataset preparation and evaluation metrics
├── requirements.txt # python dependencies
└── README.md # documentation

Dependencies

Python = 3.7
NumPy
PyTorch = 1.12.1
Open3D
hdf5storage
deepCABAC
torchac

Environment Setup

It is recommended to use a virtual environment.

conda create -n pointcloud python=3.8
conda activate pointcloud

Install PyTorch with CUDA 11.3 support.

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

Note on deepCABAC

The deepCABAC library may require manual installation and compilation depending on your environment. Please refer to the official repository for details.

https://github.com/fraunhoferhhi/deepCABAC

Testing

You can directly run test.py to evaluate the rate-distortion (RD) performance on the SemanticKITTI dataset with quantization parameters Qs = 4, 15, 66.

The test data and corresponding transformation matrices are provided in the following directories.

./test
./test_matrix

Remarks

This code is provided for research testing only. Training procedures and full datasets are not included. The complete codebase, including the training pipeline, will be released after the paper is published.
