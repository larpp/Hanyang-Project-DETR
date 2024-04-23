# CDV Detection - (Deformable) DETR
## Project
한양대학교 [STORM Lab](https://doorykimlab.wixsite.com/spectromicroscopylab)과의 공동 project로 제공받은 데이터의 detection을 해주는 역할을 맡았다.

- Cell-derived vesicle (CDV)란 Extracellular vesicle (EV)의 일종으로, Exosome과 유사하게 세포간 정보전달체 및 약물전달체로 기능할 수 있는 신개념의 나노 입자이다.

- CDV에 염색약을 뿌리면 sample에 염색약이 달라붙어 형광성을 띄게 되는데 이를 초해상력 현미경 STORM을 이용해 촬영한다.

- STORM은 0.014초마다 촬영하여 총 50,000번을 촬영해 이를 합쳐 하나의 이미지를 만들어낸다.

- 이 이미지에서 유의미한 cluster를 찾아 bounding box를 그려 box의 크기와 box안의 점들의 개수를 count한다.

- x축이 Localization number (점들의 개수), y축이 FWHM (Full Width at Half Maximum, 여기서 함수는 Gaussian 함수를 사용)인 그래프를 만든다.

- 여기서 우리의 역할은 detection model를 통해 box의 정보를 넘겨주는 것이다.

제공받은 dataset은 [Roboflow](https://universe.roboflow.com/hj-lim/cluster-3puxp)에서 확인할 수 있다.

![스크린샷 2024-03-30 145934](https://github.com/larpp/Hanyang-Project-DETR/assets/87048326/e4b83ee2-9b1d-4257-81f5-b317e7a647af)

기존에는 hand-crafted로 label 즉, box annotation을 만들어서 많은 시간과 노동이 필요했다. 인공지능을 이용해 자동으로 detection을 하는 것이 목표다. 여러 detection model을 돌려보고 서로 비교해 보았다. (Faster RCNN, DETR, Deformable DETR, YOLOv6) 여기서는 그 중 DETR과 Deformable DETR을 다뤄본다.

- [Hugging Face](https://huggingface.co/models?pipeline_tag=object-detection&sort=trending)와 [Pytorch Lightning](https://lightning.ai/docs/pytorch/1.6.0/) 1.6.0 버전을 이용했다.

- 실제 실험은 실험 환경의 동일한 setting을 위하여 [MMDetection](https://github.com/open-mmlab/mmdetection)을 활용했다.

## Model

### DETR
![스크린샷 2024-03-30 150644](https://github.com/larpp/Hanyang-Project-DETR/assets/87048326/5458d1db-617b-413b-93bf-46810d0ce851)

### Deformable DETR

![image](https://github.com/larpp/Hanyang-Project-DETR/assets/87048326/bcc4247f-45a2-4a56-baaf-a32f242e1270)

## Usage

### Environments
`nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04` 도커 이미지를 사용했다.
```
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```
Pytorch 버전은 1.10.1, python 버전은 3.8.x 버전을 사용했다.

### Data Preparation
Dataset은 COCO dataset format을 따랐다.
```
path/
  train/  # images and annotation json file
  valid/  # images and annotation json file
  test/   # images and annotation json file
```
annotation json file은 모두 _annotations.coco.json 이라는 이름으로 통일해주었다.
```
data/  # iamges
````
### Training
```
# DETR
pyhton main.py

# Deformable DETR
python main.py --checkpoint SenseTime/deformable-detr
```
### Evaluation
Test dataset의 성능을 측정한다.
```
python main.py --checkpoint <Hugging Face path> --best-map-path <ckpt path> --eval
```

### Inference
Inference할 이미지들의 결과값 (이미지, box 정보)가 저장된다.
Box 정보는 (x_center, y_center, width, height, confidence, area) 정보가 csv 파일로 저장된다.

Inference할 이미지들은 annotations가 없으므로 임의로 하나 만들어준다.
```
python predict_ann.py
```

```
python main.py --checkpoint <Hugging Face path> --best-map-path <ckpt path> --inference
```

### (Optional) Inference time
```
python benchmark.py
```
## Results
![image](https://github.com/larpp/Hanyang-Project-DETR/assets/87048326/5f9a762f-a1b0-4559-908d-ca84f39f9065)
![11-1-4](https://github.com/larpp/Hanyang-Project-DETR/assets/87048326/9c256fec-34b8-4bd8-9b86-a596ce51da79)

| Model | AP | AP50 | AP75 | AP_s | AP_m |
|:---:|:---:|:---:|:---:|:---:|:---:|
| DETR | 0.391 | 0.884 | 0.238 | 0.265 | 0.394 |
| Deformable DETR | 0.491 | 0.901 | 0.336 | 0.334 | 0.421 |
## Reference
<https://github.com/roboflow/notebooks/blob/main/notebooks/train-huggingface-detr-on-custom-dataset.ipynb>
<https://github.com/fundamentalvision/Deformable-DETR>
