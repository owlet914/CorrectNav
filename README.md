<div align="center">

# AAAI 2025 - CorrectNav 🧭

**Self-Correction Flywheel Empowers Vision-Language-Action Navigation Model**

</div>

Existing vision-and-language navigation models often deviate from the correct trajectory when executing instructions. However, these models lack effective error correction capability, hindering their recovery from errors.

To address this challenge, we propose the **Self-correction Flywheel**, a novel post-training paradigm. Instead of considering the model’s error trajectories on the training set as a drawback, our paradigm emphasizes their significance as a valuable data source. We have developed a method to identify deviations in these error trajectories and devised innovative techniques to automatically generate self-correction data for perception and action. These self-correction data serve as fuel to power the model’s continued training.

The brilliance of our paradigm is revealed when we re-evaluate the model on the training set, uncovering new error trajectories. At this time, the self-correction flywheel begins to spin. Through multiple flywheel iterations, we progressively enhance our monocular RGB-based VLA navigation model, **CorrectNav**.

## 🚀 Release Status

* [x] Release CorrectNav model weights.
* [x] Release evaluation scripts for the R2R-CE benchmark.
* [ ] Release real-world fine-tuning code (Coming Soon!).

---

## 🛠️ 1. Installation

We recommend setting up the environment on an RTX 3090 workstation with Ubuntu 22.04 and CUDA 12.1.

### 1.1 Create a Conda Environment

```bash
conda create -n CorrectNav python=3.10 cmake=3.14.0 -y
conda activate CorrectNav

```

### 1.2 Install Habitat Simulator

You will need to install specific versions of Habitat:

1. **[habitat-lab 0.3.1](https://github.com/facebookresearch/habitat-lab)**
```bash
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab  # install habitat_lab

```


2. **[habitat-sim 0.3.3](https://github.com/facebookresearch/habitat-sim/blob/main/BUILD_FROM_SOURCE.md)**
Please follow the official **Build from Source** instructions to build habitat-sim in headless mode with CUDA support.

### 1.3 Install CorrectNav Dependencies

From the root directory of this repository, run:

```bash
pip install --upgrade pip
pip install -e ".[train]"
pip install flash-attn --no-build-isolation

```

> **Note:** If you only need inference/serving, you can use `pip install -e ".[standalone]"` instead, and install extra runtime dependencies as needed.

### 1.4 Prepare the VLN-CE Dataset

Prepare the VLN datasets (R2R / RxR) by following the instructions in the [VLN-CE Data Section](https://github.com/jacobkrantz/VLN-CE?tab=readme-ov-file#data) to set up the MP3D scene dataset and VLN-CE episodes dataset.

Create a new directory named `habitat-data-0.2.5` and organize your downloaded datasets exactly as shown below:

```text
habitat-data-0.2.5/
├── datasets/
│   └── vlnnav/
│       ├── r2r/
│       │   ├── test/
│       │   ├── train/
│       │   │   ├── decompose.py
│       │   │   ├── filter.json
│       │   │   └── ...
│       │   ├── val_seen/
│       │   └── val_unseen/
│       └── rxr/
│           ├── test_challenge/
│           ├── train/
│           ├── val_seen/
│           └── val_unseen/
└── scenes/
    └── mp3d/
        ├── 17DRP5sb8fy/
        │   ├── 17DRP5sb8fy.glb
        │   ├── 17DRP5sb8fy.house
        │   ├── 17DRP5sb8fy.navmesh
        │   └── ...
        ├── 1LXtFkjw3qL/
        ├── 1pXnuDYAj8r/
        └── ...

```

---

## 📊 2. Evaluation on R2R-CE Benchmark

We provide comprehensive scripts to evaluate CorrectNav:

* **Runner:** `eval_vln_r2r_6.py`
* **Launcher:** `eval.sh`

### 2.1 Download Model Weights

📥 **[Download CorrectNav Model Weights Here](https://disk.pku.edu.cn/link/AAFD453AC93DEE4A5F8C84C14CC73D0AC1)**

### 2.2 Configuration

Before starting the evaluation, please update the evaluation scripts with your local paths and settings:

* `pretrained = "YOUR_MODEL_PATH"`
* `ckpt_chosen = ...` *(Used for naming logs and JSON outputs)*
* `CUDA_VISIBLE_DEVICES = "0..7"` *(Adjust based on your GPU availability)*

### 2.3 Run Evaluation

Start the evaluation by executing the launcher script:

```bash
bash eval.sh

```

---

## 📝 Citation

If you find our work, code, or model weights helpful in your research, please consider citing our paper:

```bibtex
@misc{correctnav,
      title={CorrectNav: Self-Correction Flywheel Empowers Vision-Language-Action Navigation Model}, 
      author={Zhuoyuan Yu and Yuxing Long and Zihan Yang and Chengyan Zeng and Hongwei Fan and Jiyao Zhang and Hao Dong},
      year={2025},
      eprint={2508.10416},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2508.10416}, 
}

```
