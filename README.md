# MultiViewStereoNet

MultiViewStereoNet is a learning-based method for multi-view stereo (MVS) depth
estimation capable of recovering depth from images taken from known, but
unconstrained, views. Unlike existing MVS methods, MultiViewStereoNet
compensates for viewpoint changes directly in the network layers. Compensating
for viewpoint changes naively, however, can be computationally expensive as the
feature layers must either be applied multiple times (once per depth
hypothesis), or replaced by 3D convolutions. We overcome this limitation in two
ways. First, we only compute our matching cost volume at a coarse image scale
before upsampling and refining the outputs. Second, we incrementally compute our
projected features such that the bulk of the layers need only be executed a
single time across all depth hypotheses. The combination of these two techniques
allows our method to perform competitively with the state-of-the-art, while
being significantly faster.

Related publication:
```
@inproceedings{greene2021mvsn,
  title={MultiViewStereoNet: Fast Multi-View Stereo Depth Estimation
         using Incremental Viewpoint-Compensated Feature Extraction},
  author={Greene, W Nicholas and Roy, Nicholas},
  booktitle={2021 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2021},
}
```
### Dependencies
This repo was developed and tested under Ubuntu 18.04 and 20.04. The following
system dependencies are required:

- python >= 3.6
- CUDA >= 10.1 (for GPU support)
- pip

All other dependencies should be installed using `pip` with the provided
`requirements.txt` file.

### Virtual Environment Setup
To run the trained models, create a virtual environment and install the `pip`
dependencies:

```bash
virtualenv -p python3 env
source ./env/bin/activate
pip3 install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

### Download Data
Scripts to download training and testing data can be found in the `scripts`
folder. You may need to install additional `pip` dependencies noted in the
respective `requirements.txt` files to extract the data. For example, to
download the GTA-SfM dataset:
```bash
cd scripts/gta_sfm

# Download data.
./download.sh

# Extract data.
virtualenv -p python3 gta_env
source ./gta_env/bin/activate
pip3 install -r requirements.txt
./extract.py
deactivate
```

### Test Network
We have provided two pre-trained networks in the `pretrained` folder: one
trained on the DeMoN dataset and one trained on the GTA-SfM dataset. To evaluate
the networks on corresponding test images and generate depth metrics run:

```bash
./test.py <weights_dir> <data_dir> <test_file>
```
The `<test_file>` should be one of the text files in the `splits` directory.

For example, to compute metrics on the 2-image GTA-SfM test set described in the paper,
run:
```bash
./test.py ./pretrained/gta_sfm_150epochs/checkpoints/epoch0149 /path/to/gta_sfm ./splits/gta_sfm_overlap0.5_test.txt
```

Other test files are provided in the `splits` directory. For example, to
evaluate depthmaps using 4 comparison images per reference image, run:
```bash
./test.py ./pretrained/gta_sfm_150epochs/checkpoints/epoch0149 /path/to/gta_sfm ./splits/gta_sfm_overlap0.5_5cmps_test.txt
```
