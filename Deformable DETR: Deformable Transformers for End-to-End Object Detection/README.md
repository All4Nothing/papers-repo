# **Deformable DETR: Deformable Transformers for End-to-End Object Detection**

- ***Link :*** https://arxiv.org/abs/2010.04159

> 💡 제안 배경 

Deformable DETR은 모델의 효율성과 수렴 속도를 개선하기 위해 제안된 모델이다. 이 논문은 DETR의 한계를 해결하고, 다양한 크기와 위치의 객체를 더 정확하게 예측할 수 있는 방법을 제시한다.

기존의 DETR은 수렴 속도가 느리고 작은 객체를 탐지하는 데 어려움이 있다. DETR이 높은 성능을 달성하기 위해 필요한 학습 시간이 상당히 긴 이유는 이미지의 모든 위치에 대해 attention을 적용하기 때문이다. 또한, DETR은 큰 해상도의 객체 탐지에는 유리하지만, 작은 객체 탐지에는 약점을 보인다.

> 💡 Deformable Attention Mechanism 

![스크린샷 2024-11-18 오후 7.26.54.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/0dbec8a6-6ae2-40e6-b117-daa31fd87a9c/e0a1dbb3-c3ac-4018-9378-032d350d2d3e/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-11-18_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_7.26.54.png)

Deformable DETR은 이미지의 모든 위치에 대해 attention을 수행하는 대신, 객체 중심의 sampling points만을 선택하여 attention을 수행한다. 이로써 연산 효율이 크게 개선되고, 모델의 수렴 속도가 향상된다.

각 쿼리 토큰은 Reference Points를 기준으로 관심 위치에 대한 offset을 학습하여 객체의 특징적인 위치만을 선택적으로 탐지한다. 이 방식을 통해 모델은 중요한 위치에만 집중할 수 있게 된다.

> 💡 Multi-Scale Feature Aggregation 

![스크린샷 2024-11-18 오후 7.26.31.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/0dbec8a6-6ae2-40e6-b117-daa31fd87a9c/725a78f5-ce16-418d-b719-775aabc530db/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2024-11-18_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_7.26.31.png)

Deformable DETR은 FPN(Feature Pyramid Network) 구조를 활용하여 다양한 크기의 객체를 효과적으로 탐지한다. 각 특징 레벨(Feature Level)의 정보를 병합함으로써 다양한 스케일의 객체 탐지가 가능해지며, 이를 통해 작은 객체에 대한 탐지 성능이 크게 향상된다.

> 💡 Reference Points 설정 방법 

Deformable DETR에서는 각 query 토큰에 대해 '참조점(reference points)'을 설정한다. 이 참조점은 이미지 내에서 query 토큰이 주목해야 할 중심 위치를 나타낸다. 이 참조점을 기준으로 deformable attention 메커니즘이 '샘플링 포인트(sampling points)'를 결정한다.

모델은 학습 과정을 통해 각 query와 가장 밀접하게 연관된다고 판단되는 이미지 내의 특정 위치로 참조점을 설정하게 된다.

> 💡 Sampling Points 설정 방법 

Deformable Attention은 각 query에 대해 고정된 개수의 sampling points를 설정한다. 이 과정에서 reference points를 기준으로 각 query마다 일정한 개수의 offset 값을 학습한다. 이 offset은 query별로 다르게 학습되어 다양한 위치의 sampling points를 결정한다. 최종적으로 reference points와 학습된 offset 값을 합산하여 sampling points를 확정한다. 이렇게 결정된 points는 주로 객체의 경계나 중요 부분을 포착하는 위치가 된다.

이렇게 결정된 sampling points에 대해 attention 연산을 수행함으로써, 중요한 위치에만 집중할 수 있게 된다. 이 과정을 통해 각 query 토큰은 중요 위치에 집중된 attention 값을 효과적으로 학습하게 된다.