## CH 4.3 데이터셋을 훈련 세트와 테스트 세트로 나누기

* 모델을 실전에 투읩하기 전에 테스트세트에 있는 레이블과 예측을 비교

  * 편향되지 않은 성능을 측정하기 위해 수행

* Wine 데이터셋

  * 178개의 와인 샘플과 여러 가지 화학 성분을 나타내는 13개의 특성으로 구성

  * ```python
    import pandas as pd
    import numpy as np
    df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
    
    df_wine.columns = ['Class label', 'Alcohol',
                      'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
                      'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 
                      'Proanthocyanins', 'Color intensity', 'Hue',
                      'OD280/OD315 of diluted wines', 'Proline']
    
    print('클래스 레이블',np.unique(df_wine['Class label']))
    df_wine.head()
    ```

  * ![fig4-3](./image/fig4-3.png)

* ```python
  from sklearn.model_selection import train_test_split
  
  X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
  ## train_test_split을 통해서 X와y를 랜덤하게 훈련세트와 테스트 세트로 분할
  ## test_size=0.3 -> X_test, y_test 가 30% X_train, y_train 가 70% 할당
  ## stratify=y -> 훈련 세트와 테스트 세트에 있는 클래스 비율이 원본 데이터셋과 동일하게 유지
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
  
  print('X_train shape : ', X_train.shape)
  print('y_train shape : ', y_train.shape)
  print('X_test shape : ', X_test.shape)
  print('y_test shape : ', y_test.shape)
  
  X_train shape :  (124, 13)
  y_train shape :  (124,)
  X_test shape :  (54, 13)
  y_test shape :  (54,)
  ```

  * 훈련 세트와 테스트 세트로 나눌 때 이 트레이드오프의 균형을 맞추어야한다.
  * 실전에서는 데이터셋의 크기에 따라 60:40, 70:30, 80:20을 사용, 대용량의 데이터셋일 경우 90:10, 99:1의 비율로 나누는 것도 보통이고 적절하다.

  *  떼어 놓았던 테스트 세트를 버리지 말고 훈련과 평가 후에 전체 데이터셋으로 모델을 다시 훈련하여 모델의 예측 성능을 향상시키는 방법이 널리 사용



## CH 4.4 특성 스케일 맞추기

* 결정트리와 랜덤 포레스트는 특성 스케일 조정에 대해 걱정할 필요가 없는 몇 안되는 머신 러닝 알고리즘 중 하나
* 경사 하강법 알고리즘을 구현하면서 보았듯이 대부분의 머신 러닝과 최적화 알고리즘은 특성의 스케일이 같을 때 훨씬 성능이 좋음.

  * 예로 들어 첫 번째 특성이 1~10사이 스케일을 가지고 잇고 두 번재 특성은 1~10만 사이 스케일을 가진다고 가정해본다면, 아달린에서 제곱오차 함수를 생각해보면 알고리즘은 대부분 두 번째 특성에 대한 큰 오차에 맞추어 가중치를 최적화할 것입니다.
  * k-최근접 이웃(KNN) 또한 샘플 간의 거리를 계산하면 두 번째 특성 축에 좌우될 것이다.

* 정규화(normalization)

  * 특성의 스케일을 [0,1] 범위에 맞추는 것

  * 최소-최대 스케일 변환(min-max scaling)의 특별한 경우

  * 데이터를 정규화 하기 위해 다음과 같이 각 특성의 열마다 최소-최대 스케일 변환을 적용하여 샘플(x^i)에서 새로운 값(x^i_norm)을 계산한다.

  * $$
    x^{(i)}_{norm} = \frac{x^{(i)} - x_{min}}{x_{max}-x_{min}}
    $$

  * 사이킷런에서 최소-최대 스케일 변환(MinMaxScaler) 클래스 제공

    ```python
    from sklearn.preprocessing import MinMaxScaler
    
    mms = MinMaxScaler()
    X_train_norm = mms.fit_transform(X_train)
    X_test_norm = mms.transform(X_test)
    ```

* 표준화(standardization)

  * 로지스틱 회귀와 SVM 같은 여러 선형 모델은 가중치를 0또는 0에 가까운 작은 난수로 초기화한다.

  * 표준화를 사용하면 특성의 평균을 0에 맞추고 표준편차를 1로 만들어 정규 분포와 같은 특징을 가지도록 만든다.

  * 이상치 정보가 유지되기 때문에 제한된 범위로 데이터를 조정하는 최소-최대 스케일 변환에 비해 알고리즘이 이상치에 덜 민감하다.

  * $$
    x^{(i)}_{std} = \frac{x^{(i)} - \mu_x}{\sigma_x}\\
    \mu_x : 어떤\; 특성의\; 샘플\; 평균\qquad \sigma_x : 그에\; 해당하는\; 표준 \;편차
    $$

  * 사이킷런에서 표준화(StandardScaler) 클래스 제공

  * ```python
    from sklearn.preprocessing import StandardScaler
    
    stdsc = StandardScaler()
    X_train_std = stdsc.fit_transform(X_train)
    X_test_std = stdsc.transform(X_test)
    ```

  

* 정규화 / 표준화 code

  * ```python
    ex = np.array([0, 1, 2, 3, 4, 5])
    print('표준화', (ex - ex.mean()) / ex.std())
    print('정규화', (ex - ex.min()) / (ex.max() - ex.min()))
    
    표준화 [-1.46385011 -0.87831007 -0.29277002  0.29277002  0.87831007  1.46385011]
    정규화 [0.  0.2 0.4 0.6 0.8 1. ]
    ```

  * ![표4-1](./image/fig4-4.png)

