# Customer Churn Analysis

**본 프로젝트는 Random Forest 알고리즘 기반의 고객 이탈 분석 및 예측기이며,**<br> 
**Python으로 설계된 데이터 로딩, 전처리, 모델 학습·평가 및 시각화 등의 기능을 포함한다.**


### 개요
본 프로젝트의 목표는 사용자의 서비스 활용 지표를 기반으로 대상의 서비스 이탈 가능성을 분석하고 예측하는 머신러닝 모델을 개발하는 것이다.<br>
학습 및 평가에 활용할 데이터셋은 [여기](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) 에서 내려받을 수 있다.

### 설치
1. 저장소 클론하기:
```sh
git clone https://github.com/alanworks72/churn-analysis.git
cd churn-analysis
```
2. 가상 환경을 생성하고 활성화하기:
```sh
python3 -m venv $가상환경명
source $가상환경명/bin/activate
```
3. 필수 패키지 설치하기:
```sh
pip3 install -r requirements.txt
```

### 사용법
1. 데이터셋을 프로젝트 디렉토리로 내려받기
1. 학습에 필요한 설정을 수정하기
    1. 데이터셋 경로, 학습 회수, 스케일러 등을 원하는 설정으로 변경하기
1. 모델 학습 스크립트 실행하기
```sh
python3 train.py
```
4. 스크립트가 종료되면 다음의 결과물이 생성됨
    1. **데이터 및 모델 성능에 대한 지표** - 터미널 출력
    1. **corr.png** - 특성 별 상관관계 히트맵
    1. **feature.png** - 특성 중요도 차트

### 성능 평가
공유된 기본 설정에 대한 모델의 성능 평가 결과
Class | Precision | Recall | F1-score | Support
-------|-----------|--------|----------|---------
0(유지)     | 0.82      | 0.91   | 0.87     | 1539
1(이탈)     | 0.67      | 0.48   | 0.56     | 574

Metric       | Score
--------------|-------
Accuracy     | 0.80
Macro Avg    | 0.71
Weighted Avg | 0.78

### 제작자
- 제작자: 박정현(alanworks72)
- 이메일: alanworks72@gmail.com
- 날짜: 2024년 7월 11일

### 라이선스
이 프로젝트는 MIT 라이선스에 따라 라이선스가 부여된다.<br>
- 자세한 내용은 LICENSE 파일 참조