# ML Seminar

## Chapter 2.1

### Ch 2.1.1 인공 뉴런의 수학적 정의

인공 뉴런 아이디어를 두개의 클래스가 있는 binary classification 작업으로 볼 수 있다.
$$
W= \begin{bmatrix} w_1 \\ \vdots \\ w_m \end{bmatrix}
,\; X= \begin{bmatrix} x_1 \\ \vdots \\ x_m \end{bmatrix}
\\
$$

$$
z = w_0x_0+w_1x_1+\cdots+w_mx_m=w^Tx
$$

$$
\phi(z)=
\begin{cases}
1 & \mbox{z}\ge 0 \mbox{ 일 때} \\
-1 & \mbox{그 외}
\end{cases}
$$

![그림2-2](./image/p051.jpg)





### Ch 2.1.2 퍼셉트론 학습 규칙

1. 가중치를 0 또는 랜덤한 작은 값으로 초기화

2. 각 훈련 샘플 x^i 에서 다음 작업 수행

   a. 출력값 계산 (출력값 = 단위 계단 함수로 예측한 클래스 레이블)

   b. 가중치 업데이트

$$
w_j:=w_j+\Delta w_j \\
\Delta w_j = \eta \left(y^{(i)} - \widehat{y}^{(i)}\right)x^{(i)}_j
$$

* 퍼셉트론이 클래스 레이블을 정확히 예측한 경우 가중치가 변경되지 않고 그대로 유지하는 반면, 잘못 예측했을 때는 가중치를 양성 또는 음성 타깃 클래스 방향으로 이동 시킨다.


$$
\Delta w_0 = \eta \left(y^{(i)} - output^{(i)} \right)\\
\Delta w_1 = \eta \left(y^{(i)} - output^{(i)} \right)x_1^{(i)}\\
\Delta w_2 = \eta \left(y^{(i)} - output^{(i)} \right)x_2^{(i)} 
$$

* 바르게 예측한 경우

$$
양성\;타깃의\;경우)\;\;\Delta w_j = \eta \left( -1 - 1 \right)x_j^{(i)}=0\\
음성\;타깃의\;경우)\;\;\Delta w_j = \eta \left( 1 - 1 \right)x_j^{(i)}=0
$$

* 잘못 예측한 경우

$$
양성\;타깃의\;경우)\;\;\Delta w_j = \eta \left( 1--1 \right)x_j^{(i)}=\eta\left(2\right)x_j^{(i)}\\
음성\;타깃의\;경우)\;\;\Delta w_j = \eta \left( -1-1 \right)x_j^{(i)}=\eta\left(-2\right)x_j^{(i)}\\
$$



![그림2-3](./image/p053.jpg)

* 퍼셉트론은 두 클래스가 선형적으로 구분되고 학습률이 충분 히작을 때만 수렴이 보장

* 두 클래스를 선형 결정 경계로 나눌 수 없다면 훈련 데이터 셋을 반복할 최대 횟수(epoch)를 지정하고 분류 허용 오차를 지정, 그렇지 않다면 가중치 업데이트를 멈추지 않는다.



![그림2-45](./image/p054.jpg)

* 샘플 x를 입력으로 받아 가중치 w를 연결하여 최종 입력을 계산
* 최종 입력은 임계 함수로 전달되어 샘플의 예측 클래스 레이블인 -1 또는 1의 이진 출력을 만듦
* 학습 단계에서 이 출력을 사용하여 예측 오차를 계산하고 가중치를 업데이트

----



## Chapter 2.2

### Ch 2.2.1 객체 지향 퍼셉트론 API

~~~python
import numpy as np

class Perceptron(object):
        # eta : learning rate
        # n_iter : epoch
        # w_ : 1d-array learning weight
        # errors_ : epoch 마다 누적된 분류 오류
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        # 훈련 데이터 학습
        # X : [n_samples, n_features]
        # y : [n_samples]
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
            
        return self
    
    def net_input(self, X):
        # 최종 입력인 z를 구현하여 return
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
~~~

* rgen.normal : 정규(가우시안) 분포에서 랜덤 표본을 추출

  ​	loc : float or array_like of floats

  ​			분포의 평균("중심")
  
  ​			Mean("centre") of the distribution

​			scale : float or array_like of floats

​						분포의 표준편차 (확산 또는 "폭")

​						Standard deviation (spread or "width") of the distribution

​			size : int or tuple of ints, optional

​			  		output shape.

​			 	 	if the given shape is e.g., (m, n, k), then m * n * k samples are drawn

​			  		if size is None(default), a single value is returned if loc ans scale are both scalars.

​			returns : ndarray or scalar

​							모수화 된 정규 분포에서 표본을 추출

​							Drawn samples from the parameterized normal distribution.



* zip() : 같은 길이의 리스트를 같은 인덱스 끼리 잘라서 리스트로 반환해주는 역할



* np.where() : list의 index를 찾는 역할

  ​		returns : ndarray

  > a=np.array([3, 4, 5, 7, 3, 1, 3])
  >
  > np.where(3==a)
  >
  > (array([0, 4, 6]), )

### Ch 2.2.2 붓꽃 데이터셋에서 퍼셉트론 훈련

* OvR ( One-versus-Rest ) / OvA ( One-versus-All)

   K개의 클래스가 존재하는 경우, 각각의 클래스에 대해 표본이 속하는지(y=1) 속하지 않는지(y=0)의 이진분류 문제를 푼다.

    클래스 수(K) 만큼의 이진 분류 문제를 풀면된다.

    판별 결과의 수가 같은 동점 문제가 발생할 수 있기 때문에 각 클래스가 얻은 조건부 확률값을 더해서 이 값이 가장 큰 클래스를 선택한다.

* OvO ( One-versus-One)

    K개의 target 클래스가 존재하는 경우, 이 중 2개의 클래스 조합을 선택하여 K(K - 1)/2 개의 이진 클래스 분류 문제를 풀고 이진판별을 통해 가장 많은 판별값을 얻은 클래스를 선택하는 방법
   
    선택 받은 횟수로 선택하면 횟수가 같은 경우도 나올 수 있기 때문에 각 클래스가 얻은 조건부 확률값을 모두 더한 값을 비교하여 가장 큰 조건부 확률 총합을 가진 클래스를 선택한다.



```python
import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                'machine-learning-databases/iris/iris.data',
                header=None)	# header=None : 칼럼 이름이 없다
df.tail()	# 데이터가 제대로 load 됬는지 확인, 마지막 5줄 출력
```

![그림 2-5](./image/fig2-5.png)



```python
import matplotlib.pyplot as plt
import numpy as np

# .iloc : numpy의 array를 인덱싱하는 것처럼 index번호로 인덱싱 
y = df.iloc[0:100, 4].values	
y = np.where(y == 'Iris-setosa', -1, 1)

X = df.iloc[0:100, [0, 2]].values

plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')

plt.xlabel('spal length [cm]')
plt.ylabel('peta length [cm]')
plt.legend(loc='upper left')
plt.show()
```

![그림 2-6](./image/fig2-6.png)

```python
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Number of errors')
plt.show()
```

![그림2-7](./image/fig2-7.png)

* 6번째 epoch 이후에 수렴했고 훈련 샘플을 완벽하게 분류하였다.



```python
def plot_decision_regions(X, y, classifier, resolution=0.02):
	# markers와 colors는 튜플로 정의
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    
    # unique(y)이므로 중복을 제외한 -1,1만 존재
    # cmap에는 colors[0], colors[1]가 매핑 (leng(unique(y)) = 2)
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 각 x축(x1), y축(x2)
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # numpy.meshgrid()는 격자의 교차점 좌표를 편하게 다룰 수 있도록 값을  리턴하는 함수
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), 
                           np.arange(x2_min,x2_max, resolution))
    # ravel() 1차원 배열로 만듦
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    # 다시 원래 배열 모양으로 복원
    Z = Z.reshape(xx1.shape)	

    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    # Z를 xx1, xx2가 축인 그래프상에 cmap을 이용해 등고선을 그림
    
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
	
    # enumerate : 순서가 있는 자료형(리스트, 튜플, 문자열)을 입력으로 받아 인덱스값을 포함하는 enumerate 				   객체를 리턴
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=cl, edgecolors='black')
        
        
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()
```

![그림2-8](./image/fig2-8.png)