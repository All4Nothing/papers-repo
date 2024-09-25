# Mask R-CNN

- ***Link :*** https://arxiv.org/abs/1703.06870

### Object Detection vs Semantic Segmentation vs Instance Segmentation  
![image](https://github.com/user-attachments/assets/8cb05032-bd5e-4197-99b7-812d103ad1d7)
![2](https://github.com/user-attachments/assets/00e52acc-a6cc-4905-9fa4-89adf1bf5bcb)

**Object Detection**

- object가 있는 곳을 bbox로 표시하며, 각 개체를 구별한다.
- B(Bounding)Box + Classification : can separate, cannot segment
- Faster R-CNN

**Semantic Segmentation**

- Semantic segmentation은 모든 픽셀에 대해 카테고리를 정한다. 개별 객체는 구분하지 않는다. 예를 들어 고양이 두마리가 있는 사진에서 두 고양이의 픽셀은 모두 ‘고양이’라는 카테고리로 분류된다. (Label each pixel in the image with a category label, Don’t differentiate instances, only care about pixels)
- Segmentation + Classification : cannot separate, can segment
- FCN은 semantic segmentation의 대표적인 방법으로, 기존에는 모든 픽셀에 대해 작은 영역으로 쪼개어 분류를 했지만, FCN

**Object Detection + Semantic Segmentation → Object Segmentation**

- Object Detection에서 가능한 separation과 Semantic Segmentation에서 가능한 segmentation을 합치자 → 픽셀단위로 객체를 구분하며, 개별 객체 또한 구분한다.
- FCN on BBOX!

### R-CNN

**R-CNN**  
![3](https://github.com/user-attachments/assets/58b71200-34c7-43ae-b9d6-3b8b54db80d5)


- R-CNN은 CNN을 object detection에 도입시킨 모델이다. R-CNN은 selective search를 사용하여 region proposal을 제공하고, CNN을 사용하여 detection을 진행한다.
- 그림과 같이 region proposal을 통해 이미지의 수많은 RoIs를 얻어낸다. 얻어진 RoIs를 CNN에 통과시켜 물체에 대한 feature vector를 추출하고, SVM을 이용하여 classification과 bounding box regression을 수행한다.

**Fast R-CNN**  
![4](https://github.com/user-attachments/assets/bfbcce47-1826-4786-8fd8-80f58a541672)

- R-CNN은 region proposal을 통해 얻어진 모든 RoIs에 대해 CNN 과정을 수행하기에 시간이 오래 걸린ㄷ. 또한 classification, CNN, region proposal이 모두 다른 stage에서 수행된다.
- Fast R-CNN은 RoI마다 CNN을 수행하지 않고, 이미지 저체에 대해 CNN을 수행한다. 그 후 생성된 feature map에 대해 region proposal과정을 거친다. 이렇게 생성된 여러 size의 RoIs를 RoI Pooling 과정을 통해 크기를 맞춘후 fully-connected layer를 통과시켜 classification과 bounding box regression을 수행한다.
- 이를 통해 CNN의 연산 횟수가 줄어 train 및 inference 수행시간이 줄고, CNN을 통해 region proposal 과정을 수행하고 그 결과를 공유하기 때문에 해당 output으로 classification과 bounding-box regression 역시 학습이 가능하게 된다.

**Faster R-CNN**  
![5](https://github.com/user-attachments/assets/d36c8d1e-ae35-4e81-afbe-4c962f0aec30)

- 하지만 Fast R-CNN은 여전히 region proposal 과정인 selective search가 CPU에서 연산이 진행되기 때문에 해당 과정에서의 속도가 느리다.


- Faster R-CNN은 region proposal 과정 또한 GPU에서 연산이 가능하도록 Region Proposal Network(RPN)을 사용하였다.

## Mask Branch

![6](https://github.com/user-attachments/assets/750f5275-cf02-42ba-843a-d64da00ac67e)

- Mask R-CNN은 Faster R-CNN에 mask branch를 추가한다.
- RoI가 input으로 들어오게 되면 FCN을 적용한 convolutional network를 통과하게 된다. 이때 마스크는 각 class마다 생성된다.
- 일반적인 FCN은 각 pixel에 해당하는 class label 값을 output으로 내는 반면, mask branch에 적용된 FCN은 물체가 존재하는지 여부에 대한 binary 값을 output으로 낸다. 따라서 기존 FCN과 다른 loss function을 사용한다.
    - Normal FCN: per-pixel softmax, multinomial cross-entropy loss
    - Mask branch FCN: per-pixel sigmoid, binary cross-entropy loss

**Mask R-CNN Loss**

![7](https://github.com/user-attachments/assets/fc27f142-ee31-4043-9807-2305d52d48fd)

- mask branch에서는 물체의 class에 따라가 아닌, 물체의 존재 여부에 따라 mask를 생성한다.

## RoIAlign

### RoI, Region of Interest

**Feature Extraction**

![8](https://github.com/user-attachments/assets/633448e5-cf83-4bf8-b69c-992f7cfbae02)

- RoI를 찾기 위해 Fast R-CNN에서는 feature map을 추출한다.
- feature map의 사이즈는 input 사이즈를 32로 나눠, 이미지 정보를 압축한다. 위의 그림의 예시에서는 512x512x3을 → 16x16x512로 압축한다.

**Get RoIs from the Feature Map**

![9](https://github.com/user-attachments/assets/77ddc1c2-4519-4a18-bbdb-77ce677a6fcc)

- 모든 RoI는 좌표와 사이즈로 이루어져 있다.

**Quantization of coordinates on the feature map**

- Quantization : process of constraining an input from a large set of values(like real numbers) to a discrete set(like integers)

![10](https://github.com/user-attachments/assets/5867dcba-4a77-4d7d-bdb8-321a99468c16)

- 사진과 같은 예시를 보면 bbox의 사이즈는 145x200이며, top-left 좌표는 (192,226)이다. 우리의 feature map은 16x16인데, 위의 200, 145와 같은 숫자는 32로 나눠떨어지지 않는다.

![11](https://github.com/user-attachments/assets/58381c6d-952b-4271-a0a7-4f5a49c64a89)

- 그래서 소수점은 버려, 결과적으로 사진의 파란색 부분의 정보를 잃게 되고, 초록색 부분에서의 새로운 정보를 얻게되며 이로인해 원래의 RoI를 사용하지 못한다.

**RoI Pooling**

![12](https://github.com/user-attachments/assets/43bb998c-b9d3-4907-abbc-e73457b0b139)
![13](https://github.com/user-attachments/assets/4d475676-c490-4f63-b9f2-e7d3d7805ba5)

- 사이즈가 다른 각 RoI들의 크기를 고정된 사이즈로 맞춰주기 위해 pooling을 진행한다. Pooling을 진행하며 그림의 마지막 행과 같이 또 한번 정보를 잃게 된다.

**RoI Align**

![14](https://github.com/user-attachments/assets/cfca0502-0715-477e-bfd3-2d5bb2b321f2)
![15](https://github.com/user-attachments/assets/20fa25ee-dd98-4faf-886c-87ecfdc4f288)
![16](https://github.com/user-attachments/assets/6dc2982f-4f55-4f07-b7b6-3bdbe1214b56)

- 기존 RoI, RoIPooling의 정보 손실 문제를 해결하기 위한 새로운 방법론이다.
- Mask R-CNN은 Instance Segmentation task를 수행하기에 pixel간의 관계가 중요하다. 따라서 정보의 손실이 없어야한다.
- RoI Align은 quantization을 하지 않고, RoI 값을 그대로 사용한다. RoI의 범위를 3x3 사이즈에 맞춰 width와 height를 3등분하고, 각 등분된 구역을 4등분하여 bilinear interpolation을 계산해 각 지점의 값을 계산할 수 있다.
- 이제 4등분된 구역의 값을 pooling하여 3x3 feature map을 생성할 수 있다.
- 결과적으로 RoI Align은 quantization없이 pooling을 진행하여 정보의 손실없이 pooling을 진행할 수 있다.

### Mask R-CNN

<img width="458" alt="17" src="https://github.com/user-attachments/assets/81fbfda7-1a3b-45be-b880-fd858f3f21b5">

- Mask R-CNN = **Faster R-CNN** with **FCN** on ROIs

**1. Mask Head on Faster R-CNN**

<img width="486" alt="18" src="https://github.com/user-attachments/assets/f6a91f4f-4932-4703-92dd-11460d610657">

Mask R-CNN branch를 제시

→ ResNet, FPN에 Mask R-CNN branch를 추가하여 사용 가능

<img width="614" alt="19" src="https://github.com/user-attachments/assets/10ab0349-4a7a-44a5-88f4-da91c5910995">

- Mask R-CNN는 Image의 feature extraction을 담당하는 backbone부분과, classification/bounding-box regression/mask prediction을 담당하는 head부분으로 나뉜다.
