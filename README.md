# coarse-to-fine-patchmatch
An implementation of Coarse-to-fine PatchMatch(CPM) Optical Flow

<img src=https://github.com/gishi523/coarse-to-fine-patchmatch/wiki/images/image1.png width=420> <img src=https://github.com/gishi523/coarse-to-fine-patchmatch/wiki/images/flow.png width=420>

## Description
- An implementation of Coarse-to-fine PatchMatch Optical Flow described in [1].
- Re-implementation of original CPM(https://github.com/YinlinHu/CPM) for OpenCV.

## References
- [1] Y. Hu, R. Song, and Y. Li. Efficient coarse-to-fine PatchMatch for large displacement optical flow. In CVPR, 2016.

## Requirement
- OpenCV
- OpenCV xfeatures2d module (for DAISY Descriptor)

## How to build
```
$ git clone https://github.com/gishi523/coarse-to-fine-patchmatch.git
$ cd coarse-to-fine-patchmatch
$ mkdir build
$ cd build
$ cmake ../
$ make
```

## How to run
```
./cpm image1 image2
```
- image1
    - the 1st input image
- image2
    - the 2nd input image

### Example
 ```
./cpm ./images/MPI-Sintel/frame_0001.png ./images/MPI-Sintel/frame_0002.png
```

## Author
gishi523
