# MultiViewSteroNet

Source code coming soon!

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
