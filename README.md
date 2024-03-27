# [FRSGM](https://doi.org/10.1109/tnnls.2023.3333538)
"Fast and Reliable Score-Based Generative Model for Parallel MRI" in IEEE TNNLS.

The training codes are in my previous projects [IDPCNN](https://github.com/Houruizhi/IDPCNN) and [TRPA](https://github.com/Houruizhi/TRPA).

To get the testing result, you can directly run the demo file.

We **Note** that the testing image must satisfy that the values in the boundary position are zero. Otherwise, the results will be unsatisfying. The examples of the testing image are in the following.

![图片1](https://github.com/Houruizhi/FRSGM/assets/43208624/ff86c8c1-22b9-418a-a912-58e966eee76d)


The checkpoint and the testing data in the paper can be found in [Google Drive](https://drive.google.com/file/d/1r_lLVGMdIFUzOLwad5YR1gt-1nYT2fFT/view?usp=sharing).

Before testing, one should use the Matlab code of ESPIRiT to estimate the sense map. 

We note that the sense map in our provided testing data is computing on the full-sampled data, with `ncalib=16`. One could also use the under-sampled data. If the number of its center lines is larger than 16, the estimated sense map is equivalent to using the full-sampled data.
