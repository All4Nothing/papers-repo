## Attention Is All You Need

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
![seq2seq](https://blog.kakaocdn.net/dn/n8ZrO/btrHofU02nK/lm59ZVTbV520tcK3ek9Adk/img.gif)  
![attention mechanism](https://blog.kakaocdn.net/dn/VFrhg/btrHn1JuCAO/U24jVZDMhen1XiU76LPBi1/img.gif)
> - Attention layer는 Attention Weight layer와 Weight Sum layer로 구성되어 있다.  
> - Attention Weight layer에서는 encoder가 출력하는 각 단어의 hidden state($h_s$)와 decoder의 LSTM layer에서 출력된 hidden state($h$)를 이용하여 가중치를 구한다. 여기서 가중치는 decoder가 예측하고자 하는 단어와 관련이 높은(유사) 정도(각 단어의 중요도)를 의미한다. 단어 간 유사 정도는 Dot-product(내적)로 구한다. Attention Weight layer에서 출력된 vector를 Attention Score라고 하며, 이 값이 클수록 두 벡터 간의 유사도가 크다는 의미이다. 일반적으로 softmax 함수(모든 가중치 합 = 1)를 적용해 normalization한다.  
> - Weight Sum layer에서는 attention weight layer에서 구한 가중치와 각 단어의 hidden state의 weighted sum(가중합)을 통해 context vector(맥락 벡터)를 출력한다.