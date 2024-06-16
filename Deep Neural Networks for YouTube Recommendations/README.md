## Deep Neural Networks for YouTube Recommendations  
***Link** : https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/45530.pdf*  
### Summary
Youtube 비디오 추천을 위한 Deep Neural Network architecutre

시스템 구조는 크게 candidate generation과 ranking 2가지로 나뉜다.

1. Candidate generation : 추천할 후보 비디오를 몇 백개 내외로 뽑아냄
- recommendation problem을 extreme multiclass classification으로 재정의
- 학습 과정에서 explict 정보(ex. ’좋아요’ 정보)는 사용하지 않고, implict 정보(비디오를 끝까지 시청했는지)만을 사용
- 트레이닝 데이터 특성상 ML을 단순하게 적용하면 오래된 아이템들이 더 추천을 많이 받게 된다. 이를 해결하기 위해 Example Age(비디오의 나이)를 input으로 넣어준다.
1. Ranking : 그 중에서 순위를 매겨 추천
- Deep Neural Network를 이용해 비디오와 사용자의 관계를 계산
