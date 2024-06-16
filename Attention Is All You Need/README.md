## Attention Is All You Need  
***Link :** https://arxiv.org/abs/1706.03762*

### Attention Mechanism

**paper**  
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)  
- [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)

**blog**  
- [[개념정리] Attention Mechanism](https://velog.io/@sjinu/개념정리-Attention-Mechanism)  
- [Attention Mechanism(어텐션 메커니즘)의 거의 모든것 (1)](https://bigdaheta.tistory.com/67)

**Summary**  
> - 기존 seq2seq는 input sequence의 길이에 상관없이 fixed-size vector로 모든 정보를 압축하기에 정보 손실이 일어나고, input sequence의 길이가 길어지면(Long-term problem), RNN에서 발생하는 vanishing gradient 문제가 발생한다. 이를 보완하기 위해 Attention Mechanism을 사용한다.    
> - Attention Mechanism이란 예측 단어(output)을 출력하기 위해, input sequence의 전부를 output과 관련이 높은 단어에 집중하여 예측에 활용한다. 즉, 중요한 부분에 더욱 Attetnion 하자는 것의 기본 컨셉이다.  
> - 기존 seq2seq모델은 encoding 단계의 마지막 hidden state만 decoder로 전달하는데, 이 경우 마지막 time-step의 정보가 더 많이 담기게 된다. 이러한 문제를 해결하기 위해 Attention의 경우, encoder의 모든 hidden state를 decoder로 전달한다.  
![seq2seq](https://github.com/All4Nothing/papers_repo/assets/81239098/09f2d447-1185-4834-a3fc-5199c627e342)
*seq2seq*
![attention mechanism](https://github.com/All4Nothing/papers_repo/assets/81239098/3828a39b-e973-4060-8cef-01ed98178571)
*Attention Mechanism*
> - Attention layer는 Attention Weight layer와 Weight Sum layer로 구성되어 있다.  
> - Attention Weight layer에서는 encoder가 출력하는 각 단어의 hidden state($h_s$)와 decoder의 LSTM layer에서 출력된 hidden state($h$)를 이용하여 가중치를 구한다. 여기서 가중치는 decoder가 예측하고자 하는 단어와 관련이 높은(유사) 정도(각 단어의 중요도)를 의미한다. 단어 간 유사 정도는 Dot-product(내적)로 구한다. Attention Weight layer에서 출력된 vector를 Attention Score라고 하며, 이 값이 클수록 두 벡터 간의 유사도가 크다는 의미이다. 일반적으로 softmax 함수(모든 가중치 합 = 1)를 적용해 normalization한다.  
> - Weight Sum layer에서는 attention weight layer에서 구한 가중치와 각 단어의 hidden state의 weighted sum(가중합)을 통해 context vector(맥락 벡터)를 출력한다.

## Abstract  
- Attention Mechanism에만 기반한 새로운 simple network architecture인 **Transformer**를 제안
- Machine Translation task에서 우월한 quality를 보여주며, 병렬처리와 학습에 있어 의미있는 시간 단축을 보여줌

## 1. Introduction  
- Recurrent 모델은 구조상 병렬 처리를 할 수 없고, 메모리 제약에 따라 sequence의 길이가 길어짐에 따라 학습에 치명적인 약점을 보임
- Attention Mechanism은 input과 output sequence의 거리에 상관없이 의존성을 모델링 할 수 있지만, 대부분의 경우 recurrent network와 함께 사용되고 있음
- Transformer는 recurrence 없이, 오로지 attention mechanism에만 의존하며 input과 output간의 전역 의존성을 모델링할 수 있는 모델임

## 2. Background  
- Extended Neural GPU, ByteNet, ConvS2S에서도 sequential computation을 줄이고자 CNN을 basic building block으로 사용해, input과 output의 hidden representations를 병렬적으로 계산함
- 이러한 모델들에서는 input과 output 사이의 관련 신호를 파악하기 위한 연산이 거리에 따라 선형 비례 혹은 로그 비례로 계산량이 늘어남
- 이에 따라 input과 output의 거리가 멀수록 의존성을 학습하기 어려움
- Transformer에서는 Multi-Head Attention을 통해 상수만큼의 계산으로 줄어듬
- Self-attention은 특정 문장 자신에게 attention을 수행해 각 단어의 표현들이 같은 문장 안에 다른 모든 단어의 표현과의 관계를 파악해 의미를 이해함
- ![image](https://github.com/All4Nothing/papers_repo/assets/81239098/76765994-c1fb-48c5-a12f-0921eca4a01d)
- 독해, 추상적 요약, 텍스트 수반, 학습 과제, 독립적인 문장 표현과 같은 task에서 성공적으로 사용됨
- Transformer는 sequence-aligned RNNs나 convolution 없이 input과 output의 표현을 계산하기 위해 self-attention에만 의존하는 첫 transduction model임

## 3. Model Architecture
- 3.1 Encoder and Decoder Stacks
- 3.2 Attention
  - 3.2.1 Scaled Dot-Product Attention
  - 3.2.2 Multi-Head Attention
  - 3.2.3 Applications of Attention in our Model
- 3.3 Position-wise Feed-Forward Networks
- 3.4 Embeddings and Softmax
- 3.5 Positional Encoding

## 4. Why Self-Attention
- Parallelization : RNN이나 CNN은 구조상 순차적으로 계산을 해야해서 병렬화에 제한이 있지만, self-attention은 각각의 위치에 대해 병렬 계산을 할 수 있어, 계산 효울성을 높이고 학습 속도를 개선할 수 있다.
- Long-range Dependencies : 기존 recurrent 모델은 sequence에서 장기간 의존성을 학습하는데 어려움을 겪는다(위 Attention Mechanism 요약 참고). 반면, self-attention은 모든 위치에서 다른 위치까지의 관계를 쉽게 학습할 수 있다.
- Interpretable : attention(가중치)를 시각화하여 sequence의 각 요소들 간의 관계를 시각화하여 볼 수 있다.
## 7. Conclusion
- attention에만 의존한 첫 sequence transduction model을 제안함
  - 흔히 사용되는 encoder-decoder 구조의 recurrent layers를 multi-headed self-attention으로 교체함
- Translation task에서 reuccrent나 convolutional layer에 기반한 구조보다 의미있는 속도 향상을 보였고, WMT 2014 English-to-Greman 과 WMT 2014 English-to-French translation task에서 SOTA 달성
- Input과 Ouptut이 text 형식뿐만이 아닌, images, audio, video 형식의 문제를 풀 수 있게 Transformer를 확장할 예정임


### 참고한 Reference
- [[Paper] Attention is All You Need 논문 리뷰](https://velog.io/@qtly_u/Attention-is-All-You-Need-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0)  
- [Attention is All You Need (Transformer)](https://velog.io/@tobigs-nlp/Attention-is-All-You-Need-Transformer)
- :star:[Attention in transformers, visually explained | Chapter 6, Deep Learning](https://youtu.be/eMlx5fFNoYc?si=xfTyAT-hOrBJXC6V):star:
