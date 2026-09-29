# STM-Net

PyTorch implementation of **Spatial-Temporal Multi-scale Network for Screen Content Video Quality Enhancement**.

**Ziyin Huang, Sik-Ho Tsang, Xinyuan Qin, Yui-Lam Chan, Xueling Zhou, and Feiyu Chen**

Ziyin Huang and Sik-Ho Tsang contributed equally to this work.

**Paper status:** Submitted; currently under major revision.

STM-Net enhances compressed screen content videos using temporal information from neighboring frames and spatial detail from the current frame. It combines a Prior-Guided Spatio-Temporal Dispatcher (PG-STD), Bidirectional Temporal Feature Extraction (BTFE), and Cascaded Multi-scale Feature Distillation (CMFD).

This repository provides the `STM_Net_NM4.py` model, training and evaluation scripts, LMDB preparation, and a QP37 checkpoint at 300,000 iterations. Only the QP37 checkpoint is included in this release.

The training entry point is the original `trainnobest.py`, renamed to `train.py`. Model, training, dataset, and evaluation logic are preserved. Executable-file edits are limited to data roots, configuration-file paths, and test-log output paths. The YAML files select the requested QP37 data and checkpoint. The LMDB-building entry point is copied unchanged from the original `HAWT/create_lmdb_mfqev2.py` and uses the existing STM-Net utilities.

## Files

```text
STM-Net/
├── STM_Net_NM4.py
├── train.py
├── test.py
├── test_ssim.py
├── create_lmdb_mfqev2.py
├── dataset/
├── utils/
├── option_R3_mfqev2_4G.yml
├── option_test_QP37.yml
├── requirements.txt
└── exp/
    └── STM_NM4_LDQP37_1920dataset_enlarge300x/
        ├── ckp_300000.pt
        ├── log.log
        └── log_test.log
```

`STM_Net_NM4.py` defines the original `STM` class with five input Y frames and 48 feature channels. The checkpoint and the two historical logs are copied without modification.

## Environment

Training and the original evaluation scripts require an NVIDIA GPU and CUDA-enabled PyTorch. Use a Python/PyTorch environment compatible with the original scripts. Dependency versions were not recorded in the source folder; `requirements.txt` lists the imported third-party packages and is not an environment lock file.

```bash
python -m pip install -r requirements.txt
```

Run commands from the `STM-Net` directory.

## Data paths

The packaged dataset roots point to `data/`. Place training YUV videos and evaluation videos as follows, or use the matching prepared QP37 LMDBs directly:

```text
data/
├── train/
│   ├── raw/
│   └── QP37/
├── scc_LD_HAWTgt37.lmdb/
│   └── meta_info.txt
├── scc_LD_HAWTlq37.lmdb/
├── eval/
│   ├── DCNraw/
│   └── DCNQP37/
└── minitest/
    ├── raw/
    └── QP37/
```

The LMDB directories also contain their database files. Training reads LMDBs using the original key/index conventions in `dataset/mfqev2.py`.

LMDB preparation, validation, and testing read the Y plane from 8-bit planar YUV444 files. Preserve the original filename convention:

```text
Ground truth: Scene_1920x1080_30_8bit_300_444.yuv
Compressed:  recScene_1920x1080_30_8bit_300_444.yuv
```

The loader obtains width, height, and frame count from the filename. Ground-truth and compressed videos must have matching dimensions and frames. Dataset files are not included in this release.

To use another data root, update the four root-directory strings in `dataset/mfqev2.py` and the data paths in the YAML files. The YAML `dataset.train.root` field is retained from the original configuration; the runtime dataset resolves its paths using `dataset/mfqev2.py`.

## Prepare training LMDBs

Run from the `STM-Net` directory after placing ground-truth YUVs in `data/train/raw/` and corresponding QP37 YUVs in `data/train/QP37/`:

```bash
python create_lmdb_mfqev2.py --opt_path option_R3_mfqev2_4G.yml
```

This creates `data/scc_LD_HAWTgt37.lmdb/` and `data/scc_LD_HAWTlq37.lmdb/`. The original helper refuses to overwrite an existing LMDB directory.

The builder uses `radius = 2`: non-overlapping groups of five LQ frames, with the third physical frame as GT. LQ keys are `im1.png` through `im5.png`; the GT key remains `im4.png`, as in the existing prepared databases. The loader reads the GT key from `meta_info.txt`, so this name does not change its physical frame alignment. Older comments mentioning seven frames or radius 3 are preserved verbatim; the executable setting is radius 2.

Ground-truth and compressed files are sorted separately and paired by their positions in those lists. Ensure the lists contain corresponding videos in the same order, with matching dimensions and frame counts. The builder processes at most the first 300 frames of each video and discards an incomplete final five-frame group. Its original helper buffers the selected Y frames as encoded images in memory before writing the database.

## Training

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --opt_path option_R3_mfqev2_4G.yml
```

The original two-GPU launcher command, with the renamed training entry point, is:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12354 train.py --opt_path option_R3_mfqev2_4G.yml
```

The original parser accepts `--local_rank`; use a PyTorch launcher version compatible with that argument. The archived training log records a two-GPU run.

Training uses the original radius of 2, 128 × 128 crops, batch size 16 per GPU, 6 workers per GPU, enlargement ratio 300, seed 7, Adam learning rate 0.0001, Charbonnier loss, and 300,000 iterations. Checkpoint saving and validation follow the original code: from iteration 200,000 at the configured 5,000-iteration interval, and at the final iteration.

The training configuration writes to `exp/STM_NM4_QP37_train/`. Use a new `train.exp_name` for each new run; the original code requires that experiment directory not to exist.

## Evaluation

Both evaluation commands select:

```text
exp/STM_NM4_LDQP37_1920dataset_enlarge300x/ckp_300000.pt
```

Y-PSNR:

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --opt_path option_test_QP37.yml
```

Y-SSIM:

```bash
CUDA_VISIBLE_DEVICES=0 python test_ssim.py --opt_path option_test_QP37.yml
```

Results are written to `log_test_current.log` and `log_test_ssim_current.log`, respectively, in the checkpoint directory. These output paths preserve the packaged historical `log_test.log`. Each script retains its original score aggregation and output format.

The included `log.log` and `log_test.log` are historical experiment records, not results produced while preparing this release.
