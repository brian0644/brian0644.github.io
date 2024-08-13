---
layout: post
title:  "ML 4주차 심화세션"
date:   2024-08-13 12:38 +09:00
categories: khuda ML session
---

# SGDClassifier loss function

## SGCClassifier의 loss = 

```python
SGDClassifier(alpha, average, class_weight, epsilon, eta0, fit_intercept, l1_ratio, 
	      learning_rat, loss, max_iter, n_iter, n_jobs, penalty, power_t, 
              random_state, shuffle, tol, verbose, warm_start)
```

SGDClassifier란 SGD(확률적 경사 하강법)을 이용한 정규화된 선형 분류 모델이다
클래스의 매개변수를 살펴보자
loss : 손실함수를 지정한다. 기본값은 "hinge" 로, 서포트 벡터 머신을 사용하는 손실함수이다
       이밖에도 log_loss, modified_huber, squared_hinge, perceptron, squared_error, huber,epsilon_insensitive, squared_epsilon_insensitive 등이 존재한다.
### 손실함수 설명
#### 1. hinge
힌지 손실은 통계적 분류 훈련에 자주 사용되는 손실 함수이다. 선형 서포트 벡터에서 대표적으로 활용되며 학습 데이터 각각의 범주를 구분하면서 데이터(서포트벡터)와의 거리(마진)가 가장 먼 결졍 경계(하이퍼플레인)를 찾는 역할을 수행한다. 
힌지 손실은 아래와 같이 정의된다.
<img width="337" alt="스크린샷 2024-08-13 오후 12 47 39" src="https://github.com/user-attachments/assets/5fd67ca8-8f6c-4d5f-84d3-8b71752f2258">

* y': 이진 분류 모델에서의 예측값
* y : 학습 데이터의 실제값 (-1 or 1)
* max{a,b} : a,b 중 더 큰 값 출력
* 둘의 부호가 같으면 1 - y'*y의 값 <= 0 즉, max{0,1-(y'x y)} = 0 >>> 잘 분류된 경우
* 둘의 부호가 다르면 1 - y'*y의 값 > 0 즉,  max{0,1-(y'x y)} = 1-(y'x y) >>> 잘못 분류 됐거나 올바르게 분류됐으나 결정 경계와 가까운 경우(0 < y'x y < 1)

** 장점 : 결정 경계를 명확히 하여 잘못 분류된 데이터 포인트에 대해 큰 벌칙을 부과. 아웃라이어에 덜 민감 **
** 단점 : 이진분류에 적합하며 다중 분류 시에는 다양한 전략이 추가될 수는 있으나 계산량이 증가할 수 있음. 특성이 많을때 성능 저하 가능성. 스케일이 다른 데이터를 전처리하지않으면 잘못된 결정 경계 도출(like knn 예제) **
