---
layout: post
title:  "ML 4주차 심화세션"
date:   2024-08-13 12:38 +09:00
categories: khuda ML session
---

# 4주차 심화세션 - SGDClassifier loss function

## SGDClassifier의 loss : 이론

```python
SGDClassifier(alpha, average, class_weight, epsilon, eta0, fit_intercept, l1_ratio, 
	      learning_rat, loss, max_iter, n_iter, n_jobs, penalty, power_t, 
              random_state, shuffle, tol, verbose, warm_start)
```

SGDClassifier란 SGD(확률적 경사 하강법)을 이용한 정규화된 선형 분류 모델이다

 * alpha : 값이 클수록 강력한 규제 설정 (default = 0.0001)
 * loss : 손실함수 (default = 'hinge')
 * epsilon : 손실 함수에서 현재 예측과 올바른 레이블 간의 차이가 임계값보다 작으면 무시 (default = 0.1)
 * penalty : 규제 종류 선택 (default = 'l2', 'l1', 'elasticnet'(l1,l2 섞은거))
 * l1_ratio : L1 규제의 비율 (Elastic-net에서만 사용 , default = 0.15)
 * max_iter : 계산에 사용할 작업 수


  
>>> 클래스의 매개변수 중 loss에 들어가는 다양한 손실함수에 대해 살펴보겠다

  
loss : 손실함수를 지정한다. 기본값은 "hinge" 로, 서포트 벡터 머신을 사용하는 손실함수이며
       이밖에도 log, modified_huber, squared_hinge, perceptron, squared_error, huber,epsilon_insensitive, squared_epsilon_insensitive 등이 존재한다.

       
### 손실함수 설명
#### 1. hinge, squared_hinge
  힌지 손실은 통계적 분류 훈련에 자주 사용되는 손실 함수이다. 선형 서포트 벡터에서 대표적으로 활용되며 학습 데이터 각각의 범주를 구분하면서 데이터(서포트벡터)와의 거리(마진)가 가장 먼 결졍 경계(하이퍼플레인)를 찾는 역할을 수행한다. 
squard_hinge는 해당 함수에 제곱을한 형태이다

  
<img width="337" alt="스크린샷 2024-08-13 오후 12 47 39" src="https://github.com/user-attachments/assets/5fd67ca8-8f6c-4d5f-84d3-8b71752f2258">

  
<img width="337" alt="스크린샷 2024-08-14 오전 03 30 02" src="https://github.com/user-attachments/assets/e4d776b2-07b9-4154-89f6-039168115f78">

* y': 이진 분류 모델에서의 예측값
* y : 학습 데이터의 실제값 (-1 or 1)
* max{a,b} : a,b 중 더 큰 값 출력
* 둘의 부호가 같으면 1 - y'*y의 값 <= 0 즉, max{0,1-(y'x y)} = 0 >>> 잘 분류된 경우
* 둘의 부호가 다르면 1 - y'*y의 값 > 0 즉,  max{0,1-(y'x y)} = 1-(y'x y) >>> 잘못 분류 됐거나 올바르게 분류됐으나 결정 경계와 가까운 경우(0 < y'x y < 1)
* -> squared_hinge는 그 값들을 제곱함으로 잘못 분류된 경우 큰 오차를 부여, 올바르게 분류됐으나 결정 경계와 가까운 경우 더 작은 손실을 부과한다.


  
**장점 : 결정 경계를 명확히 하여 잘못 분류된 데이터 포인트에 대해 큰 벌칙을 부과. 아웃라이어에 덜 민감**

  
**단점 : 이진분류에 적합하며 다중 분류 시에는 다양한 전략이 추가될 수는 있으나 계산량이 증가할 수 있음. 특성이 많을때 성능 저하 가능성. 스케일이 다른 데이터를 전처리하지않으면 잘못된 결정 경계 도출(like knn 예제)**

  
#### 2. log
  로그 손실은 로지스틱 회귀에서 사용하는 로지스틱 손실 함수이다. 
  이진분류(시그모이드)와 다중분류(소프트맥스) 상황에서 모델의 예측값과 실제 레이블 간의 로그 손실을 계산하고 손실이 최소화되도록 가중치를 조정하여 모델을 학습시킨다.
  로그 손실은 아래와 같이 정의된다.


  <img width="452" alt="스크린샷 2024-08-13 오후 1 24 33" src="https://github.com/user-attachments/assets/5d69fa60-ec1d-4dbc-a398-596f2c4064fb">

  <img width="313" alt="스크린샷 2024-08-13 오후 1 24 20" src="https://github.com/user-attachments/assets/aa976259-e2f4-4b79-8ce2-83164b0cc76e">

* N : 샘플의 수
* K : 클래스의 수
* y : 학습 데이터의 실제 값
* 𝑦̂ : 예측된 확률 값

**장점 : 각 타깃에 대한 확률값을 제공하기 때문에 확률적 해석이 가능. 다중 클래스 분류 문제에서도 사용 가능**

  
**단점 : 잘못된 예측에 대해 큰 패널티를 부과 -> 손실값이 매우 커져 모델에 큰 영향**


#### 3. huber, modified_huber
  우선, 후버 손실은 MSE(평균 제곱 오차)와 MAE(평균 절댓값 오차)를 결합한 함수이다.
  MSE는 이상치에 제곱을 통해 많은 비중을 주게되는 반면, MAE는 같은 비중을 두게된다. 이러한 점에서 MSE의 경우 이상치가 아닌 데이터에선 훌륭한 모델을 만들 수 있기에 이 둘을 절충한 것이 후버 손실이다. 주로 회귀 문제에 사용한다.
  반면, 수정된 후버 손실은 주로 이진 분류 문제에서 사용되며 이진 분류 문제에서 로그 손실의 단점을 보완하기 위해 등장하였다. 로그 손실은 잘못된 예측에 큰 패널티를 부과한다는 점에서 학습을 방해할 수 있으며 이 문제를 해결하고자 나온 것이다.
  후버 손실과 수정된 후버 손실은 아래와 같이 정의된다.

  
  <img width="356" alt="스크린샷 2024-08-13 오후 1 47 26" src="https://github.com/user-attachments/assets/6ede9b54-3e9d-4eec-8419-0596b1a34ba8">
<img width="511" alt="스크린샷 2024-08-13 오후 1 52 06" src="https://github.com/user-attachments/assets/792fdeb1-eb05-428d-ae35-e44334b8a356">


* e : 에러 (예측값과 실제값의 차이)
* 델타 : MSE와 MAE를 나누는 분기점
* 에러의 절댓값이 델타보다 작을 경우 MSE 적용, 다른 모든 지점에 대하여 MAE 적용

**장점 : 노이즈가 많거나 이상치가 존재하는 데이터셋에 유리하다**
  
  
**단점 : 매개변수 델타 선택에 어려움이 있다. 이 임계값을 적절히 설정하는 것이 매우 중요하다**


#### 4. perceptron
  퍼셉트론 손실 함수는 간단한 선형 분류 알고리즘인 퍼셉트론에서 사용하는 손실함수로, 힛지 손실함수와 유사하다. max(0,-y x f(x))일 뿐

    
<img width="511" alt="스크린샷 2024-08-14 오전 3 38 42" src="https://github.com/user-attachments/assets/fb49d83c-9408-4a56-bff7-dcd57cb2ab9e">


* y : 실제 데이터의 값
* f(x) : 예측값
* 둘의 부호가 같으면 - y*f(x)의 값 <= 0 즉, max{0,-(y'x y)} = 0 >>> 잘 분류된 경우
* 둘의 부호가 다르면 - y*f(x)의 값 > 0 즉, >>> 잘못 분류 됐거나 올바르게 분류됐으나 결정 경계와 가까운 경우

  
**장점 : 계산이 간단하고 이해하기 쉬우며 선형 분리가 가능한 데이터에 대해 빠르게 수렴한다 + 메모리 사용이 적다.**
  
  
**단점 : 노이즈나 선형 분리가 불가능한 데이터에 대해 성능이 떨어지며 이상치에 민감하다.**




  


## SGCClassifier의 loss : 실습

### 1. 분석 환경 준비

```python
import pandas as pd
run_walk = pd.read_csv('/Users/brian0644/Desktop/python/khuda/run_or_walk.csv')
run_walk.columns
```
![스크린샷 2024-08-13 오후 11 48 16](https://github.com/user-attachments/assets/65b42e01-59a7-45c9-9abb-4ce69a0b468e)


  -> accelerration은 각 축에 대한 각가속도, gyro는 3차원 공간에서 각 축에 대한 각속도를 의미. 

### 2. 데이터 전처리

```python
# 훈련세트와 테스트세트 분리
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

x = run_walk.drop(['date', 'time', 'username', 'activity'], axis=1)     # 시간, 이름, 종속변수 제거하고 남는 데이터를 데이터로
y = run_walk['activity']

trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2)


# 훈련세트와 테스트세트의 데이터(not 타깃) 스케일 조정
scaler = StandardScaler()
scaler.fit(trainX)
trainX = scaler.transform(trainX)
testX = scaler.transform(testX)
```

### 3. loss function 5개 비교

```python
# 필요한 패키지 임포트
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
import numpy as np



# loss function들을 리스트에 넣고 반복문 진행, scores에 점수 기록
losses = ["hinge", "log_loss", "modified_huber", "perceptron", "squared_hinge"]
scores = []
for loss in losses:
    clf = SGDClassifier(loss=loss, penalty="l2", max_iter=1000)
    clf.fit(trainX, trainY)
    scores.append(clf.score(testX, testY))
  
plt.title("Effect of loss")
plt.xlabel("loss")
plt.ylabel("score")
x = np.arange(len(losses))
plt.xticks(x, losses)
plt.plot(x, scores) 

```

### 4. 결과 해석
![스크린샷 2024-08-14 오전 12 05 26](https://github.com/user-attachments/assets/a997594e-467f-49c4-bc9f-e77132194798)


* 반복적으로 결과를 수행해 보아도 유독 perceptron 손실함수를 사용한 모델은 다른 손실함수에 비해 점수가 낮다
* 이러한 결과는 데이터의 분포가 선형 분리로 정확하게 나누기 어려운 상태임을 추측할 수 있다


