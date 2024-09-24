# Distance Ratio Based Adjuster for Animeinterp

> **Abstract：** This project serves as a control mechanism for Video Frame Interpolation (VFI) networks specifically
> tailored for anime.
> By calculating the DistanceRatioMap, it adjusts the frame interpolation strategies for spatiotemporally nonlinear and
> linear regions,
> thereby preserving the original pace and integrity of the characters while avoiding distortions common in frame
> interpolation.

# 👀Demo

## input
![input](https://github.com/hyw-dev/FCLAFI/assets/68835291/cc9fb083-0f8d-48e1-b33e-0a893f313329)
## output
![output](https://github.com/hyw-dev/FCLAFI/assets/68835291/5138f267-6904-42ce-9551-b0891812a650)

# 👀Demos Videos

**[OP「つよがるガール」](https://www.bilibili.com/video/BV1uJtPe9EdY/?share_source=copy_web&vd_source=8a8926eb0f1d5f0f1cab7529c8f51282)**

**[葬送的芙莉莲 NCOP1「勇者」](https://www.bilibili.com/video/BV12QsaeREmr/?share_source=copy_web&vd_source=8a8926eb0f1d5f0f1cab7529c8f51282)**

**[葬送的芙莉莲 NCOP2「放晴」](https://www.bilibili.com/video/BV1RYs8eFE77/?share_source=copy_web&vd_source=8a8926eb0f1d5f0f1cab7529c8f51282)**


## 🔧Dependencies
**Set up the environment for the following repository**
- [Practical-RIFE](https://github.com/hzwer/Practical-RIFE)
- [FastFlowNet](https://github.com/ltkong218/FastFlowNet)
- [GMFSS](https://github.com/98mxr/GMFSS_Fortuna)

## ⚡Usage 

**Coming soon.**

# 📖Version Comparison

## gmfss(For learning and reference only.)

## rife_lite(Aimed at real-time playback.)

**Due to the limitations of the RIFE algorithm's performance, some distortion may occur in the background during
compensation.**

![rife](https://github.com/user-attachments/assets/e0480165-c748-43ac-ad3c-5e6fb7adea7f)

**If you use FastFlowNet TRT, RIFE v4.22 Lite TRT, and implement the project with VapourSynth, real-time playback can be
achieved on a reasonably powerful NVIDIA GPU, such as the RTX 3070.**

## gmfss_union(For high-quality output.)

**Combining the strengths of both RIFE and GMFSS, it delivers outstanding results.**

![gmfss](https://github.com/user-attachments/assets/5a4ca540-ddfa-4a93-ab21-e39eb9299e89)
