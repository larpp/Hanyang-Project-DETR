# CDV Detection - DETR
## Project
- 한양대학교 [STORM Lab](https://doorykimlab.wixsite.com/spectromicroscopylab)과의 공동 project로 제공받은 데이터의 detection을 해주는 역할을 맡았다.

- Cell-derived vesicle (CDV)란 Extracellular vesicle (EV)의 일종으로, Exosome과 유사하게 세포간 정보전달체 및 약물전달체로 기능할 수 있는 신개념의 나노 입자이다.

- 이를 초해상력 현미경 STORM을 이용해 촬영하고 이미지 파일로 제공받고 DETR 모델을 이용해 detection 해보았다.

- 제공받은 dataset은 [Roboflow](https://universe.roboflow.com/hj-lim/cluster-3puxp)에서 확인할 수 있다.

![스크린샷 2024-03-30 145934](https://github.com/larpp/Hanyang-Project-DETR/assets/87048326/e4b83ee2-9b1d-4257-81f5-b317e7a647af)

## Model
![스크린샷 2024-03-30 150644](https://github.com/larpp/Hanyang-Project-DETR/assets/87048326/5458d1db-617b-413b-93bf-46810d0ce851)
Detection model은 DETR로 선택했다.

## Environments
`nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04` 도커 이미지를 사용했다.
```
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```
Pytorch 버전은 1.10.1을 사용했다.

## Data Preparation
