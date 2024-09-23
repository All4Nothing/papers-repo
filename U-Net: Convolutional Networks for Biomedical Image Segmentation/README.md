# **U-Net: Convolutional Networks for Biomedical Image Segmentation**

- ***Link :*** https://arxiv.org/abs/1505.04597

**💡 Model Architecture**

![IMG_1262](https://github.com/user-attachments/assets/a30d5067-fb66-46b2-b00c-88ca60edd437)  
모델의 구조가 U모양이라 U-Net
Contracting Path와 Expanding Path로 구성되어 있다.

**Contracting Path**

- Downsampling 과정을 반복하며 Feature Map을 생성한다.
- 주변 픽셀들을 참조하는 범위를 넓혀가며 이미지로부터 Contextual 정보를 추출한다.
- 3x3 Convolution을 수행할 때 padding을 하지 않아 feature map의 크기가 감소한다.
- Downsampling할 때 마다 channel의 수를 2배 증가시키며 진행한다.(1 → 64 → 128 → 256 → 512 → 1024)

**Expanding Path**

- Upsampling 과정을 반복하여 Feature Map을 생성한다.
- Skip Connection을 통해 contracting path에서 생성된 contextual 정보와 위치정보를 결합한다.
  

**💡 Improved Sliding Window Search Methoed**

![IMG_1263](https://github.com/user-attachments/assets/4d753613-d5e0-4562-ad38-31c89aced94e)  
기존의 Sliding Window Search Method 기법은 윈도우를 조금씩 이동하면서 물체를 탐색하기에 겹치는 부분이 존재 → 이전 윈도우에서 검증이 끝난 부분을 한번 더 검증하면서, 시간과 연산 측면에서 낭비가 발생함
U-Net은 검증이 끝난 곳은 검증하지 않음. 이미 검증한 부분은 건너뜀



**💡 Overlap tile Method (Strategy)**

![IMG_1264](https://github.com/user-attachments/assets/9cb1273e-ea2b-4686-bdf6-8e71d7758008)  
U-Net은 padding을 진행하지 않고, convolution을 진행하기에 output의 크기가 input 크기보다 작음.
→ Overlap tile method를 사용
- missing value는 mirroring 기법을 사용함
- 경계를 기준으로 value를 대칭하여 missing value를 채움

  

**💡 Data Augmentation**. 

Biomedical 분야는 특히나 labeld 데이터가 부족해서 data augmentation이 중요
![IMG_1265](https://github.com/user-attachments/assets/00d37a16-7e6a-48ac-b0d5-9d9fdacc0ea0)  
이 논문에서는 위와 같은 기본적인 data augmentation이 아닌 Elastic Deformation(탄성변형)을 이용. 이미지를 픽셀 별로 일정한 확률을 가지고 변형되기에, 좀 더 현실에서 있을 법한 변화를 보임 → Biomedical에 더 적합 (세포도 살아있어 모습이 순간순간마다 다를 수 있음)


  
**💡 Training**  
U-Net은 SGD를 이용해서 학습.
적은 batch 사이즈를 사용함으로써 생기는 단점(최적화 문제, 적은 샘플을 참고함) → 모멘텀 값을 크게 하여 과거의 값이 좀 더 많이 반영되게 학습

**Weight Loss**
모델이 객체간 경계를 구분할 수 있도록 Weight Loss를 구성하고 학습한다.
![IMG_1266](https://github.com/user-attachments/assets/7c9558f9-eb51-419f-b5c8-9cb1a1222bfd)  
이미지 경계를 잘 학습함을 볼 수 있음

### Reference
- https://joungheekim.github.io/2020/09/28/paper-review/
- https://medium.com/@msmapark2/u-net-논문-리뷰-u-net-convolutional-networks-for-biomedical-image-segmentation-456d6901b28a
- https://youtu.be/O_7mR4H9WLk?si=-KbUNDf9Ayj11gLu
