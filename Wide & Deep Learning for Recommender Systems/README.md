## Wide & Deep Learning for Recommender Systems
***Link** : https://arxiv.org/abs/1606.07792*  

### Summary  
- Recommender system에서는 Memorization과 Generalization 모두 중요하다.
- Wide Model : Memorization된 값을 통해서 추천 (ex. 아아를 검색한 사용자가 쿠키를 시킨 기록이 많았다면, 아아를 검색한 사용자에게 쿠키를 추천하게 됨) → 기존에 기억된 결과로만 추천
- Deep Model : 아이템을 일반화 시켜서 추천 (ex. 아아를 일반화(커피)하여, 라떼를 추천) → 지나치게 일반화(under filtering)되는 문제가 발생할 수 있음(ex. 아아를 검색한 사람에게 따뜻한 라떼를 추천)
- 두 모델을 결합한 Wide & Deep Model 제안
- Google Play store에서 높은 효율 개선을 보임
