"""
    Author: Alan Park (alanworks72@gmail.com)
    Date: Jul.10.2024
    Modified: Jul.11.2024
    File Name: data.py
    Version: 1.1
    Description: Data loader for Churn analyzer
"""

import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import seaborn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

""" FEATURE INFO
        customerID: Customer ID
        gender: Whether the customer is male or a female
        SeniorCitizen: Whether the customer is a senior citizen or not (1,0)
        Partner: Whether the customer has a partner or not (Yes,No)
        Dependents: Whether the customer has dependents or not (Yes,No)
        tenure: Number of months the customer has stayed with the company
        PhoneService: Whether the customer has a phone service or not (Yes,No)
        MultipleLines: Whether the customer has multiple lines or not (Yes,No,No phone service)
        InternetService: Customer's internet service provider (DSL,Fiber optic,No)
        OnlineSecurity: Whether the customer has online security or not (Yes,No,No internet service)
        OnlineBackup: Whether the customer has noline backup or not (Yes,No,No internet service)
        DeviceProtection: Whether the customer has device protection or not (Yes,No,No internet service)
        TechSupport: Whether the customer has tech support or not (Yes,No,No internet service)
        StreamingTV: Whether the customer has streaming TV or not (Yes,No,No internet service)
        StreamingMovies: Whether the customer has streaming movies or not (Yes,No,No internet service)
        Contract: The contract term of the customer (Month-to-month,One year,Two year)
        PaperlessBilling: Whether the customer has paperless billing or not (Yes,No)
        PaymentMethod: The customer's payment method (Electronic check,Mailed check,Bank transfer(automatic),Credit card(automatic))
        MonthlyCharges: The amount charged to the customer monthly
        TotalCharges: The total amount charged to the customer
        Churn: Whether the customer churned or not (Yes,No)
"""

def dataloader(path):
    """
    @brief CSV 파일을 로드하고 전처리하는 함수
    
    @param path(str): CSV 파일 경로
    
    @return x_train(np.array): 학습 데이터
    @return x_test(np.array): 테스트 데이터
    @return y_train(pd.Series): 학습 데이터 레이블
    @return y_test(pd.Series): 테스트 데이터 레이블
    """
    df = pd.read_csv(path)

    # 데이터 타입 확인
    print(df.dtypes)

    # TotalCharges 열을 수치형으로 변환
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # 결측값 확인
    print(df.isnull().sum())
    print(df.isna().sum())
    
    # 결측값 처리
    df = df.fillna({"TotalCharges": df['TotalCharges'].median()})

    # 범주형 변수 인코딩
    le = LabelEncoder()
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = le.fit_transform(df[column])

    # 특성 간 상관관계 시각화
    plt.figure(figsize=(12, 10))
    seaborn.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.yticks(rotation=45, ha="right")
    plt.xticks(rotation=30, ha="right")
    plt.savefig("./corr.png")

    # 특성과 레이블 분리
    x = df.drop("Churn", axis=1)
    y = df["Churn"]

    # 데이터셋 분할 - 학습:테스트 = 7:3
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test