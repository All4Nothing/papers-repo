## SESSION-BASED RECOMMENDATIONS WITH RECURENT NEURAL NETWORKS

- ***Link :*** https://arxiv.org/abs/1511.06939

## Summary

> RNN을 이용한 session-based recommendation. 
User의 개별 session에서 발생하는 행동을 기반으로 개인화된 추천을 제공할 수 있으며, RNN의 sequential한 특징 덕분에 user의 관심사가 시간에 따라 변하는 경우에도 유연하게 대응할 수 있어 기존의 방식보다 더 정확한 추천을 제공할 수 있다.
> 

### session-based recommendation

- user는 웹사이트나 앱을 사용하는 동안 여러 아이템을 시청하거나 여러 아이템과 상호 작용한다(클릭, 구매 등). 이 session 동안 발생하는 사용자의 행동 패턴을 기반으로 아이템을 추천한다.

## Abstract

- recommender systems에 RNN을 적용
- 현실에서 추천 시스템은 long user histories(e.g. Netflix)보다 short session-based data(e.g. 작은 쇼핑몰 사이트)를 바탕으로 추천해야 하는 문제가 있다.
- short-based 상황에서는 Matrix Factorizaiton은 정확도가 떨어진다. → 데이터가 너무 없기 때문
- 이런 문제는 주로 item-to-item recommendations을 이용해 비슷한 아이템을 추천하는 방식으로 이용하지만, user based가 아니기에 추천 정확도가 상대적으로 떨어짐 → user가 A라는 아이템을 봤을 때, A 아이템과 비슷한 아이템들만 추천하는 경향 발생
- 이 논문에서는 전체 session을 모델링하면 더 정확한 추천을 제공할 수 있을 것이라고 생각한다.
- 따라서 session-based recommendations에 RNN 접근 방식을 제안한다.
- 또한, RNN의 ranking loss function을 변경하여 session-based recommendations task에서 좋은 성능을 낼 수 있었다.

## 1. Introduction

- Session-based recommendation은 ML과 추천 시스템 커뮤니티에서 상대적으로 진가를 인정받지 못했다.
- 대부분의 e-커머스를 위한 session-based recommendation system은 user profile을 만들지 않는 상대적으로 simple한 methods를 사용한다. 예를 들어, item-to-item similarity, co-occurence, transition probabilities.
- 주로 사용하는 factor model은 user profile이 없기 때문에 session-based 추천 문제를 해결할 수 없지만, Neighborhood method는 similiarity와 co-occurrence 기반이기 때문에 session-based 추천에서 적절히 사용 가능했음
- RNN은 순서가 있는 데이터를 모델링 하기에 적합한 구조라서 (RNN의 sequential한 특징) session의 시간적 흐름을 고려하여 RNN을 활용할 수 있음
    - initial input : user가 웹사이트에 들어와서 하는 첫 번째 클릭
    - output: 각각의 이전 클릭과 이어지는 다음 클릭

## 3. RECOMMENDATIONS WITH RNNs

- Vanishing gradient 문제를 보완하기 위해 GRU를 사용. → GRU는 gate를 통해 hidden state를 언제, 얼만큼 업데이트할지 학습한다.

### 3.1.3 Ranking Loss

- Pointwise ranking : 각 item에 점수를 매긴다.
- Pairwise ranking : user가 선택한 item과 선택하지 않은 item을 고르고, user가 선택한 item을 선택하지 않은 item보다 더 선호할 확률을 비교한다. 두 값들의 차이가 커지도록 한다.
- Listwise ranking :  모든 item에 score와 rank를 부여하고 정답과 비교한다.
    
    → Pairwise ranking이 성능이 제일 좋았음. BPR(Bayesian Personalized Ranking), TOP1을 사용했다.
    

### 5. CONCLUSION

- Session-based Recommendation with Recurrent Neural Network(GRU)를 제안
- Session-parallel mini-batches, Mini-batch based output sampling, Ranking Loss function 제안