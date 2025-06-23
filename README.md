# KICLIP: Knowledge Injection Improves Distillation

<!-- pdf based image -->
![KICLIP](./fig/heatmap_comp.png)

<strong>Figure 1</strong>: EigenCAM comparison between the model without Knowledge Injection (top row) and with Knowledge Injection (bottom row). The injected
knowledge helps the student model better align with the teacher’s internal representations, as evident from the improved feature structure. Visualizations are
shown for every transformer layer after the 5th layer in ViT-B/16. Red areas indicate most-attended regions. This figure qualitatively illustrates the effect of Knowledge Injection on internal feature representations. The EigenCAM comparison shows that intermediate knowledge injection clearly improves the model’s ability to attend to the subject in the frame. Without Knowledge Injection, the model struggles to focus on the subject even by the 10th layer. In contrast, when Knowledge Injection is used, the model begins attending to the subject as early as the 7th layer. Furthermore, in the final layer, the attention is noticeably more focused and structured around the subject, indicating better alignment with the teacher model.

<!-- ![KICLIP](./fig/KICLIP.png) -->

## How to Use

<p style="font-weight:bold; color:red">First clone this repository</p>
<p style="font-weight:bold; color:red">Then enter the directory of this repo in your terminal</p>

## Dependencies

```bash
pip install scipy
pip install pandas
pip install scikit-learn
pip install ftfy
pip install regex
```

```bash
pip install 'git+https://github.com/facebookresearch/fvcore'
pip install simplejson
conda install av -c conda-forge
pip install -U iopath
pip install psutil
pip install opencv-python
pip install tensorboard
git clone https://github.com/facebookresearch/pytorchvideo.git
cd pytorchvideo
pip install -e .
git clone https://github.com/facebookresearch/detectron2 detectron2_repo
pip install -e detectron2_repo
```

## Data Preparation

- **Kinetics-400.**

  We obtained the compressed version Kinetics-400 dataset, where videos have been resized to 256, from the [`VoV3d Repo`](https://github.com/youngwanLEE/VoV3D/blob/main/DATA.md#kinetics-400). The repository provides the download link for the dataset: [`Kinetics-400 dataset link`](https://dl.dropbox.com/s/419u0zljf2brsbt/compress.tar.gz). After downloading and extracting the data, you should rename the folders "train_256" and "val_256" to "train" and "val" respectively. Additionally, please note that the video "val/crossing_river/ZVdAl-yh9m0.mp4" is invalid and needs to be replaced. You should download a new version of the video from [`here`](https://drive.google.com/file/d/15M07kKQlZEoVzUezppITSnICs83fch8A/view?usp=share_link) and perform the replacement.

- **UCF-101.**

  We download UCF-101 dataset by the [`script`](https://github.com/open-mmlab/mmaction2/blob/main/tools/data/ucf101/download_videos.sh) provided by MMAction2.

- **HMDB-51.**

  We download the HMDB-51 dataset by the [`script`](https://github.com/open-mmlab/mmaction2/blob/main/tools/data/hmdb51/download_videos.sh) provided by MMAction2.

- **Kinetics-600 testing.**

  The validation data of Kinetics-600 we used can be downloaded from [`link`](https://pan.baidu.com/s/1d6wI-n3igMdE1rJ2xP2MsA?pwd=c5mu).

The data should be in the following structure:

```bash
ROOT PROJECT DIRECTORY
- data
    - k400
        - train
        - val
        - test
    - ucf101
    - hmdb51
        - videos
    - kinetics-600
        - val
```

## Training

<p style="font-weight:bold; color:red">Please inspect the scripts to change the paths according to the arrangements on your system.</p>

#### Base To Novel Setting

1. `bash train_b2n_hmdb51.sh`
2. `bash train_b2n_ucf101.sh`
3. `bash train_b2n_k400.sh`

#### Cross Dataset Setting

Released soon...

## Evaluation

You need to use the weight average tool. Make sure to change the directories according to the experiment in the weight_average_tool.py file. Then run:
`python weight_average_tool.py`

#### Base To Novel Setting

Run the test scripts. Make sure to change the directories according to the experiment in the test scripts.

`bash test_b2n_hmdb51.sh`
`bash test_b2n_ucf101.sh`
`bash test_b2n_k400.sh`

#### Cross Dataset Setting

Released soon...
