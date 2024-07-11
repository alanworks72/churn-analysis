"""
    Author: Alan Park (alanworks72@gmail.com)
    Date: Jul.11.2024
    File Name: train.py
    Version: 1.0
    Description: Model trainer for Churn analyzer
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from data import dataloader


def train(dataset):
    """
    @brief 모델을 학습시키고 성능을 검증하는 함수
    
    @param dataset(tuple): 학습과 테스트를 위한 데이터셋 (x_train, x_test, y_train, y_test)
    """
    x_train, x_test, y_train, y_test = dataset
    features = list(x_train.keys())

    # 데이터 스케일링
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Random Forest Classifier 학습
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    # 모델 테스트
    y_pred = model.predict(x_test)

    # 모델 성능 검증
    validation(y_test, y_pred)
    # 학습 결과 시각화
    visualize(model, features)

def validation(y_test, y_pred):
    """
    @brief 모델의 성능을 검증하는 함수
    
    @param y_test(np.array): 실제 레이블
    @param y_pred(np.array): 예측된 레이블
    """
    print("Model Accuracy:\n", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

def visualize(model, features):
    """
    @brief 모델의 특성 중요도를 시각화하는 함수
    
    @param model(sklearn.ensemble.RandomForestClassifier): 학습된 모델
    @param features(list): 특성 이름 리스트
    """
    # 특성 별 중요도 추출
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # 특성 중요도 시각화
    plt.figure(figsize=(10,6))
    plt.title("Feature Importances")
    plt.bar(range(len(features)), importances[indices], align="center")
    plt.xticks(range(len(features)), [features[i] for i in indices], rotation=20, ha="right")
    plt.savefig("./feature.png")

def run(path):
    """
    @brief 데이터 로드 후 모델 학습을 시작하는 함수
    
    @param path(str): 데이터셋 파일 경로
    """
    dataset = dataloader(path)
    train(dataset)


if __name__ == "__main__":
    path = "./Telco-Customer-Churn.csv"
    run(path)