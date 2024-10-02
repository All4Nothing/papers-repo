# **End-to-End Object Detection with Transformers**

- ***Link :*** https://arxiv.org/abs/2005.12872

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/0dbec8a6-6ae2-40e6-b117-daa31fd87a9c/6579cba2-10d2-4e71-90fd-02085a4b339a/image.png)

<aside>
💡

이 논문의 장점은?

</aside>

기존의 객체 탐지(Object detection) 기술과 비교했을 때 매우 간단하며 또한 경쟁력 있는 성능을 보인다.

<aside>
💡

이 논문이 제안한 것은?

</aside>

DEtection TRansformer, DETR은 Bipartite matching loss function(이분 매칭 손실 함수)를 제시했고, Transformer를 이용한 object detection task를 수행한다.

1. CNN을 backbone network로 사용하여 이미지의 features를 추출한다.
2. 이미지에서 상대적인 위치 정보를 담기 위해 positional encoding을 추가하여 encoder로 들어간다.
3. encoding된 정보가 decoder로 들어간다.
4. 각 query가 decoder에 들어가 나온 output이 각각 FFN을 거쳐 class와 bbox가 출력된다.

## Bipartite Matching

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/0dbec8a6-6ae2-40e6-b117-daa31fd87a9c/be5d3bb2-777b-48bd-8d86-76532bd87bff/image.png)

기존 object detection 방법들은 너무 복잡하며 다양한 라이브러리를 활용한다. 또한, bounding box의 형태, bounding box가 겹칠 때의 처리 방법과 같은 prior knowledge(사전 지식)가 요구된다. 예를 들어, 탐지하고자 하는 object가 기차와 같이 긴 물체일 경 bounding box를 길게 설정한다던가 하는 것이다.

이 논문에서는 이러한 문제를 bipartite matching(이분 매칭)을 통해 set prediction problem을 직접적으로 해결한다. 여기서 set은 수학에서 말하는 집합이다. 집합은 중복되는 원소가 없고, 원소의 순서 또한 상관이 없다.

이미지에서 탐지할 object의 개수를 고정해두면, 이분 매칭을 수행할 수 있다.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/0dbec8a6-6ae2-40e6-b117-daa31fd87a9c/f762ba4e-ce32-4a85-93dd-a5a8e60a8968/image.png)

이때, loss가 가장 작게 나오도록, 예측 결과와 가장 비슷한 실제 값을 매칭한다.

## Transformer

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/0dbec8a6-6ae2-40e6-b117-daa31fd87a9c/99f852ec-5550-4c7d-adca-0124fdf6cf0b/image.png)

### Encoder

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/0dbec8a6-6ae2-40e6-b117-daa31fd87a9c/af14fbfa-7232-4f24-bda2-08247d809639/image.png)

- Encoder는 $d \times HW$크기의 연속성을 띠는 feature map을 입력으로 받는다. 이때 $d$는 image featuref를 의미하고 $HW$는 각각의 픽셀 위치 정보를 담고 있다.
- Encoder의 self-attention map을 시각화해보면 개별 인스턴스를 적절히 분리하는 것을 확인할 수 있다.

### Decoder

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/0dbec8a6-6ae2-40e6-b117-daa31fd87a9c/d7612b97-ac03-4f61-8426-054967870182/image.png)

- Decoder는 $N$개의 object query(학습된 위치 임베딩)를 초기 입력으로 이용한다. 인코더가 global attention을 통해 인스턴스를 분리한 뒤에 디코더는 각 인스턴스의 클래스와 경계선을 추출한다.