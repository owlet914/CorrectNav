**CorrectNav**
=========

Existing vision-and-language navigation models often deviate from the correct trajectory when executing instructions. However, these models lack effective error correction capability, hindering their recovery from errors. To address this challenge, we propose the Self-correction Flywheel, a novel post-training paradigm. Instead of considering the model’s error trajectories on the training set as a drawback, our paradigm emphasizes their significance as a valuable data source. We have developed a method to identify deviations in these error trajectories and devised innovative techniques to automatically generate self-correction data for perception and action. These self-correction data serve as fuel to power the model’s continued training. The brilliance of our paradigm is revealed when we re-evaluate the model on the training set, uncovering new error trajectories. At this time, the self-correction flywheel begins to spin. Through multiple flywheel iterations, we progressively enhance our monocular RGB-based VLA navigation model, CorrectNav.
You can see our paper in [CorrectNav](https://arxiv.org/abs/2508.10416) and more demo videos on our [homepage](https://correctnav.github.io/).

> Notes
>
> - The VLN / Habitat pipeline requires installing **habitat-sim** and **habitat-lab** and preparing the corresponding datasets.
> - Several research scripts in this repo contain hard-coded local paths. For a new environment, you should update those paths.

## 1) Installation

### 1.1 Create an environment

```bash
conda create -n CorrectNav python=3.10 -y
conda activate CorrectNav
```

### 1.2 Install Habitat and datasets

Install:

- habitat-sim **0.3.3**
- habitat-lab **0.3.1**

and prepare the VLN datasets (R2R / RxR) following upstream instructions:

- habitat-sim: https://github.com/facebookresearch/habitat-sim
- habitat-lab: https://github.com/facebookresearch/habitat-lab

### 1.3 Install CorrectNav dependencies

From the repo root:

```bash
pip install --upgrade pip
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

> If you only need inference/serving, you can start from `pip install -e ".[standalone]"` and then install extra runtime deps as needed.

## 2) Evaluate   Our    Model

### 5.1 VLN-CE evaluation scripts (R2R)

You can download our model [here](https://disk.pku.edu.cn/link/AAFD453AC93DEE4A5F8C84C14CC73D0AC1).
This repo provides evaluation runners:

- `eval_vln_r2r_6.py`

and a launcher script: `eval.sh`.

Before running, update the evaluation scripts:

- `pretrained = "YOUR MODEL PATH"`
- `ckpt_chosen = ...` (used for naming logs / json outputs)

Then run:

```bash
bash eval.sh
```

`eval.sh` currently launches 8 GPU workers (`CUDA_VISIBLE_DEVICES=0..7`). Adjust the loop if you have a different GPU count.



## BibTex

Please cite our paper if you find it helpful :)

```
@misc{yu2025correctnavselfcorrectionflywheelempowers,
      title={CorrectNav: Self-Correction Flywheel Empowers Vision-Language-Action Navigation Model}, 
      author={Zhuoyuan Yu and Yuxing Long and Zihan Yang and Chengyan Zeng and Hongwei Fan and Jiyao Zhang and Hao Dong},
      year={2025},
      eprint={2508.10416},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2508.10416}, 
}
```

