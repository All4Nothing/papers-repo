# SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers

- ***Link :*** https://arxiv.org/abs/2105.15203
  

> 💡 SegFormer의 연구 배경

과거의 Segmentation은 주로 두 가지 방향으로 연구가 진행되었다. 하나는, 이미지의 특징을 추출하기 위한 Encoder, 즉 Backbone 모델을 개선하여 성능을 향상시키는 방향. 또는, 이미지 내의 문맥 정보를 효율적으로 추출하기 위한 방법 추가하여 성능 향상시키는 방향으로 진행되었다.

이후 ViT가 등장하였고, ViT를 Encoder로, CNN을 Decoder로 사용하는 SETR이 좋은 성능을 냈지만, 몇가지 문제가 존재하였다.

먼저, ViT는 CNN과 달리 하나(single-scale)의 저해상도 특징들만을 사용한다. CNN은 convolution과 pooling 연산을 통해 하나의 이미지에 여러 해상도의 feature를 사용하는 것과 달리 ViT는 고정된 크기의 해상도만을 feature로 사용한다. 두번째 문제로는, 이미지 크기에 따라 연산량이 크게 증가한다는 문제이다. ViT는 동일한 크기의 패치를 사용하기에 이미지의 크기가 커지면, 패치가 그만큼 늘어나게 되고, 모든 패치들과의 유사도를 계산하는 self-attention 매커니즘의 특성상 연산량이 제곱배로 늘어난다.

이러한 문제들을 해결하기 위해 PVT, Swin Transforemr, Twins 등 여러 방법론들이 등장하였지만, 주로 Encoder에 대해 다룰 뿐 Decoder에 대해 다루지는 않았다.
  

> 💡 Segformer의 주요한 특징

이 논문에서 제시한 Segformer의 주요한 특징은 다음과 같다.

먼저, Hierarchical Transformer Encoder를 사용하여 다양한 scale의 특징들을 사용할 수 있다. 

또한, Positional Encoding을 사용하지 않았기에 학습에 사용한 이미지와 크기가 다른 이미지가 들어와도 추론 성능이 크게 감소하지 않았다.

마지막으로, 간단한 구조의 MLP를 Decoder로 사용하여, 더 적은 연산량을 가지면서도 Encoder가 추출한 모든 특징들을 잘 활용할 수 있다.
  

> 💡 Segformer의 모델 구조

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/0dbec8a6-6ae2-40e6-b117-daa31fd87a9c/e3f40ed9-bfeb-49bf-8b4b-55b59c9e6fb0/Untitled.png)

1. 입력 이미지를 4x4 크기의 패치로 나눈다.
2. Encoder에서 이를 Hierarchical Transformer의 입력으로 넣어 원본 이미지의 1/4, 1/8, 1/16, 1/32 크기의 feature map을 얻는다.
3. Decoder에서는 Encoder로 얻어낸 모든 feature map을 MLP에 넣어 최종 output을 출력한다.
  

> 💡 Hierarchical Transformer Encoder

본 논문에서는 Hierarchical Transformer Encoder를 Mix Transformer(MiT)로 이름 붙였다.

**Hierarchical Feature Representation**

MiT는 동일한 수의 Patch를 토대로 연산을 진행하며 Patch의 수가 변하지 않는다. 

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/0dbec8a6-6ae2-40e6-b117-daa31fd87a9c/11c308f8-128a-4dc1-ab6b-f712404b941d/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/0dbec8a6-6ae2-40e6-b117-daa31fd87a9c/840dab20-41b9-4f0f-a198-df1157613d98/Untitled.png)

이 구조는 CNN과 유사한 형태로 고해상도의 Coarse한 특징들과 저해상도의 Fine-Grained 특징들을 얻어 segmentation에서 더욱 좋은 성능을 낼 수 있다.

**Overlapped Patch Merging**

또한, 기존의 ViT 계열의 모델에서 사용하는 Patch Merging 방법 대신 Overlapped Patch Merging을 사용한다.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/0dbec8a6-6ae2-40e6-b117-daa31fd87a9c/b6840c99-7034-4d0b-ae79-946b77ec2114/Untitled.png)

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/0dbec8a6-6ae2-40e6-b117-daa31fd87a9c/b821470b-2400-4b2d-8947-5ed0ec0ffc39/Untitled.png)

기존 Patch Merging 방법은 인접한 Patch들을 붙이는 방법이었기에 다른 부분으로 병합된 패치들과의 정보는 단절된다.

이를 해결하기 위해 고안된 Overlapped Patch Merging은 다른 패치와의 정보를 교환할 수 있도록 한다. 

Conv 연산과 비슷하게 K(Kernel size), S(Stride), P(Padding)을 정의하여 비슷한 원리로 Patch를 병합한다. 

본 논문에서는 (K, S, P)를 각각 (7, 4, 3), (3, 2, 1)를 사용하였다.

**Efficient Self-Attention**

Encoder의 Self-Attention의 연산량으로 인해 bottleneck 현상이 발생한다. 기존 Attention 연산에서 패치의 수가 $N$이라면 시간 복잡도는 $O(N^2)$
이 된다. 이를 해결하기 위해 Reduction Ratio $R$을 사용하여 시간 복잡도를 줄인다.

그 과정은 다음과 같다.

$\hat{K}=Reshape(\frac{N}{r},C\cdot R), \ K=Linear(C\cdot R, C)(\hat{K})$

이를 통해 Attention의 시간 복잡도를 $O(N^2)$에서 $O(\frac{N^2}{R})$로 줄일 수 있다.

본 논문에서는 $R$을 stage-1부터 stage-4까지 각각 64,16,4,1를 사용하였다.

**Mix-FFN**

기존 ViT는 Positional Encoding을 사용해 각 패치에 위치 정보를 제공하였다. 이로 인해, 모델의 학습에 사용된 이미지와 다른 해상도의 이미지가 들어올 경우 성능이 크게 감소되는 결과가 나왔다.

본 논문에서는 이를 해결하기 위해 Mix-FFN을 소개한다.

Mix-FFN은 기존의 Feed Forward Network에 3x3 Conv 연산을 적용한다.

$MixFFN(x)=MLP(GELU(Conv_{3\times3}(MLP(x))))+x$

본 논문에서는 이 3x3 Conv 연산을 통해 Positional Encoding을 충분히 대체할 수 있다고 한다.

또한, Depth-Wise Conv를 사용해 parameters 수를 줄이고 효율성을 증가시켰다.
  

> 💡 Lightweight All-MLP Decoder

Segformer에서는 기존의 다른 방법들과 달리 매우 경량화된 Decoder를 사용하면서도 좋은 성능을 낸다.

그 이유는 Hierarchical Transformer Encoder가 CNN보다 넓은 Effective Receptive Field(ERF)를 가지고 있기 때문이다.

Decoder의 수식은 다음과 같다.

$\hat{f}=Linear(C_i,C)(F_i) \\ \hat{f}=Upsample(\frac{W}{4} \times \frac{W}{4})(\hat{F}_i) \\ F=Linear(4C,C)(Concat (\hat{F}_i)) \\ M=Linear(C,N_{cls})(F)$

이때 $\hat{f}_i$는 각 Encoder의 출력이며 $c_i$는 각 Encoder의 채널 수이다.

**Effective Receptive Field Analysis**

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/0dbec8a6-6ae2-40e6-b117-daa31fd87a9c/cfb9745c-ee72-46a0-aad5-fe9a370652ac/Untitled.png)

DeepLab V3+ 모델과 Segformer의 ERF를 시각화한 모습을 보면, DeepLab V3+ 모델의 ERF가 Segformer보다 상대적으로 작음을 확인할  수 있다.

Segformer의 Receptive Field를 보면 빈 부분이 없이 골고루 인식함을 알 수 있고, 이로 인해 Encoder만으로도 Global Context 또한 잘 인식할 수 있다. 또한, 이로인해 간단한 Decoder 하나만으로도 넓은 Receptive Field를 가진다.

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/0dbec8a6-6ae2-40e6-b117-daa31fd87a9c/508010dc-1816-4029-9a01-c658d2a5a31b/Untitled.png)