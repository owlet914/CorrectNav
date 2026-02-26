This is a fantastic project. Structuring a README effectively is crucial for maximizing academic impact and making it easy for the robotics community to reproduce your work.

I have reorganized your repository document to make it highly scannable and professional. I added standard repository badges, a dedicated **Release Status** section with the checkboxes you requested, and stylized the formatting to highlight key terms.

Here is the beautified Markdown code for your README:

---

```markdown
# CorrectNav: Self-Correction Flywheel Empowers Vision-Language-Action Navigation Model

[![arXiv](https://img.shields.io/badge/arXiv-2508.10416-b31b1b.svg)](https://arxiv.org/abs/2508.10416)
[![Project Page](https://img.shields.io/badge/Project-Website-blue)](https://correctnav.github.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



Existing vision-and-language navigation (VLN) models often deviate from the correct trajectory when executing instructions. However, these models lack effective error correction capability, hindering their recovery from errors. 

To address this challenge, we propose the **Self-correction Flywheel**, a novel post-training paradigm. Instead of considering the model’s error trajectories on the training set as a drawback, our paradigm emphasizes their significance as a valuable data source. We have developed a method to identify deviations in these error trajectories and devised innovative techniques to automatically generate self-correction data for perception and action. These self-correction data serve as fuel to power the model’s continued training. 

The brilliance of our paradigm is revealed when we re-evaluate the model on the training set, uncovering new error trajectories. At this time, the self-correction flywheel begins to spin. Through multiple flywheel iterations, we progressively enhance our monocular RGB-based VLA navigation model, **CorrectNav**.

---

## 🚀 Release Status

We are continuously updating this repository with new features and code. 

- [x] **Model Weights:** Open-sourced the pre-trained CorrectNav weights.
- [x] **Evaluation Scripts:** Released code to evaluate CorrectNav on the R2R-CE Benchmark.
- [ ] **Real-World Fine-Tuning:** Plan to release the code and deployment scripts for real-world robotic fine-tuning.

---

## 🛠️ 1. Installation

### 1.1 Create an Environment
We have tested the following installation process on an RTX 3090 workstation running Ubuntu 22.04 and CUDA 12.1.

```bash
conda create -n CorrectNav python=3.10 cmake=3.14.0 -y
conda activate CorrectNav

```

### 1.2 Install Habitat Simulator

Install the required Habitat dependencies:

**Habitat-Lab (0.3.1)**

```bash
git clone --branch stable [https://github.com/facebookresearch/habitat-lab.git](https://github.com/facebookresearch/habitat-lab.git)
cd habitat-lab
pip install -e habitat-lab  # install habitat_lab

```

**Habitat-Sim (0.3.3)**
Please follow the [Build from Source](https://github.com/facebookresearch/habitat-sim/blob/main/BUILD_FROM_SOURCE.md) instructions to build `habitat-sim` in headless mode with CUDA support.

### 1.3 Install CorrectNav Dependencies

From the root of this repository, run:

```bash
pip install --upgrade pip
pip install -e ".[train]"
pip install flash-attn --no-build-isolation

```

> **Note:** If you only need inference/serving capabilities, you can start with `pip install -e ".[standalone]"` and then install extra runtime dependencies as needed.

### 1.4 Prepare the VLN-CE Dataset

Prepare the VLN datasets (R2R / RxR) by following the instructions in the [VLN-CE Data Section](https://github.com/jacobkrantz/VLN-CE?tab=readme-ov-file#data) to set up the MP3D scene dataset and VLN-CE episodes dataset.

Create a new folder named `habitat-data-0.2.5` and place all scene data in `scenes/` and all episode data in `datasets/`. Organize your directory exactly like this:

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

## 📊 2. Evaluate CorrectNav on R2R-CE Benchmark

We provide the following scripts for seamless evaluation:

* **Runner:** `eval_vln_r2r_6.py`
* **Launcher:** `eval.sh`

### Download Model Weights

Download the pre-trained CorrectNav model weights from [PKU Disk](https://disk.pku.edu.cn/link/AAFD453AC93DEE4A5F8C84C14CC73D0AC1) and place them in your designated model directory.

### Configuration

Before initiating the evaluation, please update `eval.sh` and the evaluation scripts with your local paths and environment settings:

```python
pretrained = "YOUR_MODEL_PATH"
ckpt_chosen = "YOUR_CHECKPOINT_NAME" # Used for naming logs and JSON outputs
CUDA_VISIBLE_DEVICES = "0..7"

```

### Run Evaluation

Start the evaluation process by executing the launcher script:

```bash
bash eval.sh

```

---

## 📖 Citation

If you find CorrectNav or our self-correction flywheel paradigm helpful in your research, please consider citing our paper:

```bibtex
@misc{correctnav,
      title={CorrectNav: Self-Correction Flywheel Empowers Vision-Language-Action Navigation Model}, 
      author={Zhuoyuan Yu and Yuxing Long and Zihan Yang and Chengyan Zeng and Hongwei Fan and Jiyao Zhang and Hao Dong},
      year={2025},
      eprint={2508.10416},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={[https://arxiv.org/abs/2508.10416](https://arxiv.org/abs/2508.10416)}, 
}

```

```

***

As you prepare to open-source the real-world fine-tuning code, would you like me to help draft a specialized hardware section detailing your physical robot deployment instructions (e.g., configuring stereo depth cameras or specifying manipulator control modes)?

```
