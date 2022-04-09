# Two View Image Matcher

## About

This is a simple two view image matcher for finding the correspondences between the two view images, estimating the extrinsics (rotation and translation), and warping one image to another image space. The inputs are query rgb image, query depth information, query camera intrinsic parameters, target rgb image, target rgb depth, and target camera intrinsic parameters. This project is in C++ and requires opencv libraries:

1. opencv xfeatures2d library to find SURF correspondences and check their validity using Lowe's match filtering and the depth information provided.

2. opencv calib3d library to estimate Essential Matrix with 5-point algorithm. Then decompose Rotation Matrix and Translation Matrix from it. Finally, concatenate them to Extrinsic Matrix.

3. opencv calib3d library to estimate Homography Matrix to form the perspective transformation between the two view images. Then warp the query image on the target image with the perspective warping.

## Data

Please use the following naming convention to save the data in a same folder.

1. A Target example [RGB: target.jpeg](data/target.jpeg), [depth: target.depth.tiff](data/target.depth.tiff), [intrinsic: target.json](data/target.json)

2. A good Query example respect to the Target [RGB: good.jpeg](data/good.jpeg), [depth: good.depth.tiff](data/good.depth.tiff), [intrinsic: good.json](data/good.json)

3. A bad Query example respect to the Target [RGB: bad.jpeg](data/bad.jpeg), [depth: bad.depth.tiff](data/bad.depth.tiff), [intrinsic: bad.json](data/bad.json)

## Instruction

1. After compiling the main.cpp, run the binary with three inputs: cwd, target, and query.

```bash
./image_match --cwd ~/data/ --target target --query good
```

which you save the [RGB: target.jpeg](data/target.jpeg), [depth: target.depth.tiff](data/target.depth.tiff), [intrinsic: target.json](data/target.json), [RGB: good.jpeg](data/good.jpeg), [depth: good.depth.tiff](data/good.depth.tiff), and [intrinsic: good.json](data/good.json) in [folder: ~/data/](data/)

2. The program will print out Intrinsic Matrices, Essential Matrix, Extrinsic Matrix, Homography Matrix in the console, and finally display the warped image of the query projected onto the target. Press any key to save the warped image in cwd, the directory where saved the query and target image with the image name query_warp.jpeg. (e.g. [warp: good_warp.jpeg](data/good_warp.jpeg))
