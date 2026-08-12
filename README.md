# Distance Ratio Based Adjuster for Animeinterp

> **Abstract：** This project serves as a control mechanism for Video Frame Interpolation (VFI) networks specifically
> tailored for anime.
> By calculating the DistanceRatioMap, it adjusts the frame interpolation strategies for spatiotemporally nonlinear and
> linear regions,
> thereby preserving the original pace and integrity of the characters while avoiding distortions common in frame
> interpolation.


<a href="https://colab.research.google.com/drive/1BGlSg7ghPoXC_s5UuF8Z__0YV4fGrQoA?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="google colab logo"></a>
### 📘[中文文档](README_CN.md)  

# 👀Demo
https://github.com/user-attachments/assets/1f1dd01a-2edb-4198-a4a8-edf0979bb8ba


## 🔧Installation

```bash
git clone https://github.com/routineLife1/DRBA.git
cd DRBA
pip3 install -r requirements.txt
```
The cupy package is included in the requirements, but its installation is optional. It is used to accelerate computation. If you encounter difficulties while installing this package, you can skip it.

## ⚡Usage 

**Video Interpolation**
```bash
  # For speed preference
  python infer.py -m rife -i input.mp4 -o output.mp4 -fps 60 -scale 1.0 -s -st 0.3
  # For quality preference
  python infer.py -m gmfss_union -i input.mp4 -o output.mp4 -fps 60 -scale 1.0 -s -st 0.3
```

**Full Usage**
```bash
Usage: python infer.py -m model -i in_video -o out_video [options]...
       
  -h                   show this help
  -m model             model name (rife, gmfss, gmfss_union) (default=rife)
  -i input             input video path (absolute path of input video)
  -o output            output video path (absolute path of output video)
  -fps dst_fps         target frame rate (default=60)
  -t times             interpolation times (default=-1, if specified, the times mode will be used as priority)
  -s enable_scdet      enable scene change detection (default False)
  -st scdet_threshold  ssim scene detection threshold (default=0.3)
  -hw hwaccel          enable hardware acceleration encode (default Disable) (require nvidia graph card)
  -scale scale         flow scale factor (default=1.0), generally use 1.0 with 1080P and 0.5 with 4K resolution
```

- model accept model name. Current support: rife, gmfss, gmfss_union
- input accept absolute video file path. Example: E:/input.mp4
- output accept absolute video file path. Example: E:/output.mp4
- dst_fps = target interpolated video frame rate. Example: 60
- times = interpolation times. Example: 2 (if specified, the times mode will be used as priority)
- enable_scdet = enable scene change detection.
- scdet_threshold = scene change detection threshold. The larger the value, the more sensitive the detection.
- hwaccel = enable hardware acceleration during encoding output video.
- scale = flow scale factor. Decrease this value to reduce the computational difficulty of the model at higher resolutions. Generally, use 1.0 for 1080P and 0.5 for 4K resolution.

**Using the VapourSynth + TensorRT version can achieve 400% acceleration. Refer to this project: [VS-DRBA](https://github.com/routineLife1/VS-DRBA).**

# 👀Other Demos(BiliBili)

**[Sousou no Frieren NCOP1](https://www.bilibili.com/video/BV12QsaeREmr/?share_source=copy_web&vd_source=8a8926eb0f1d5f0f1cab7529c8f51282)**

**[Sousou no Frieren NCOP2](https://www.bilibili.com/video/BV1RYs8eFE77/?share_source=copy_web&vd_source=8a8926eb0f1d5f0f1cab7529c8f51282)**

**[OP「つよがるガール」](https://www.bilibili.com/video/BV1uJtPe9EdY/?share_source=copy_web&vd_source=8a8926eb0f1d5f0f1cab7529c8f51282)**

## 📖Overview
DRBA consists two parts('DRM Calculation' and 'Applying DRM to Frame Interpolation') to generate the adjusted in-between anime frame given three inputs.
![Overview](assert/Overview.png)

# 🔗Reference
Optical Flow: [GMFlow](https://github.com/haofeixu/gmflow)

Video Interpolation: [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) [GMFSS](https://github.com/98mxr/GMFSS_Fortuna) [MultiPassDedup](https://github.com/routineLife1/MultiPassDedup)

## Acknowledgment
This project is sponsored by [SVFI](https://steamcommunity.com/app/1692080) [Development Team](https://github.com/Justin62628/Squirrel-RIFE) 