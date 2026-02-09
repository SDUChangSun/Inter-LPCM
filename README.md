# Inter-LPCM

Inter-LPCM is the official research code for a **learning-based point cloud compression framework**.  
The system integrates **point cloud registration**, **deep neural networks**, and **entropy coding** to efficiently compress 3D point clouds.  

> ⚠️ This code is for **testing and experimental evaluation only**.

---

## Dependencies

- Python 3.7
- NumPy
- PyTorch 1.12.1
- Open3D
- hdf5storage
- deepCABAC
- torchac

---

## Environment Setup

It is recommended to use a **virtual environment**:

```bash
# Create and activate conda environment
conda create -n pointcloud python=3.8
conda activate pointcloud

# Install PyTorch with CUDA 11.3 support
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 \
    --extra-index-url https://download.pytorch.org/whl/cu113
```

Note on deepCABAC
deepCABAC may require manual installation and compilation depending on your environment.
Refer to the official repository for details: https://github.com/fraunhoferhhi/deepCABAC

## Testing

Run test.py to evaluate rate-distortion (RD) performance on the SemanticKITTI dataset.
Supported quantization parameters: Qs = 4, 15, 66 , has set in the code.


Test data and transformation matrices are located in:

./test
./test_matrix

## Remarks

This code is provided for research testing only.

The full codebase, including the training pipeline and entropy model with more bitrate, will be released after the paper is published.
