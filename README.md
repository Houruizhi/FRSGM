# [FRSGM](https://doi.org/10.1109/tnnls.2023.3333538)
"Fast and Reliable Score-Based Generative Model for Parallel MRI" in IEEE TNNLS.

The training codes are in my previous projects [IDPCNN](https://github.com/Houruizhi/IDPCNN) and [TRPA](https://github.com/Houruizhi/TRPA).

To get the testing result, you can directly run the demo file.

We **Note** that the testing image must satisfy that the values in the boundary position are zero. Otherwise, the results will be unsatisfying. The examples of the testing image are in the following.

![图片1](https://github.com/Houruizhi/FRSGM/assets/43208624/ff86c8c1-22b9-418a-a912-58e966eee76d)


The checkpoint and the testing data in the paper can be found in [Google Drive](https://drive.google.com/file/d/1ThVsaKe2SfY0z_RNxPfPy3q8sOeGSglW/view?usp=sharing).

Before testing, one should use the Matlab code of ESPIRiT to estimate the sense map. Or you can use the open source library in Python tools to estimate the sense map.
```python
import sigpy.mri as mr
mps = mr.app.EspiritCalib(
                    torch.view_as_complex(data_kspace).cpu().numpy(),
                    calib_width=12,
                    thresh=0.04,
                    kernel_width=4,
                    crop=0.97,
                    max_iter=100
                ).run()
sense_map_tensor = torch.view_as_real(torch.from_numpy(mps)).to(data_kspace.device)
```

If the code is helpful, please cite the following papers:
```
@article{HOU2022113973,
title = {IDPCNN: Iterative denoising and projecting CNN for MRI reconstruction},
journal = {Journal of Computational and Applied Mathematics},
volume = {406},
pages = {113973},
year = {2022},
author = {Ruizhi Hou and Fang Li},
}

@ARTICLE{TRPA,
  author={Hou, Ruizhi and Li, Fang and Zhang, Guixu},
  journal={IEEE Transactions on Computational Imaging}, 
  title={Truncated Residual Based Plug-and-Play ADMM Algorithm for MRI Reconstruction}, 
  year={2022},
  volume={8},
  number={},
  pages={96-108}
}


@ARTICLE{FRSGM,
  author={Hou, Ruizhi and Li, Fang and Zeng, Tieyong},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Fast and Reliable Score-Based Generative Model for Parallel MRI}, 
  year={2023},
  volume={},
  number={},
  pages={1-14}
}
```
