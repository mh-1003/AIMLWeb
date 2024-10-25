import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# 타이틀과 간단한 설명 추가
st.title("Iris Species Predictor")
st.write("""         -------         """)

# 데이터 로드 및 데이터프레임 생성
iris = load_iris()
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in y]

# 데이터프레임 표시
st.subheader('Iris Dataset')
st.write(df)

# 입력 슬라이더 추가
st.sidebar.header('Input Parameters')
def user_input_features():
    sepal_length = st.sidebar.slider('sepal length(cm)', 
                                     float(df['sepal length (cm)'].min()),
                                     float(df['sepal length (cm)'].max()),
                                     float(df['sepal length (cm)'].mean()))
    sepal_width = st.sidebar.slider('sepal width(cm)', 
                                    float(df['sepal width (cm)'].min()), 
                                    float(df['sepal width (cm)'].max()), 
                                    float(df['sepal width (cm)'].mean()))
    petal_length = st.sidebar.slider('petal length(cm)', 
                                     float(df['petal length (cm)'].min()), 
                                     float(df['petal length (cm)'].max()), 
                                     float(df['petal length (cm)'].mean()))
    petal_width = st.sidebar.slider('petal width(cm)', 
                                    float(df['petal width (cm)'].min()), 
                                    float(df['petal width (cm)'].max()), 
                                    float(df['petal width (cm)'].mean()))
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# 사용자 입력 데이터 표시
st.subheader('User Input parameters')
st.write(input_df)

# 모델 학습
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# 예측
prediction = model.predict(input_df.to_numpy())
prediction_proba = model.predict_proba(input_df.to_numpy())

st.subheader('Prediction')
st.write(iris.target_names[prediction])
st.write('Prediction Probability')
st.write(prediction_proba)

# 데이터 시각화
st.subheader('Iris Dataset Visualization')
sns.pairplot(df, hue='species')
st.pyplot(plt)

# 히스토그램
st.subheader('Histogram')
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for i, ax in enumerate(axes):
    sns.histplot(df[iris.feature_names[i]], kde=True, ax=ax)
plt.tight_layout()
st.pyplot(fig)

# 상관행렬
st.subheader('Correlation Matrix')
plt.subplots(figsize=(10, 8))
numerical_df = df.drop('species', axis=1)
corr_matrix = numerical_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.tight_layout()
st.pyplot(fig)

# 페어플롯
st.subheader('Pairplot')
fig = sns.pairplot(df, hue='species').fig
plt.tight_layout()
st.pyplot(fig)

# 피처 중요도 시각화
st.subheader('Feature Importance')
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
fig, ax = plt.subplots()
ax.bar(range(X.shape[1]), importances[indices], align="center")
ax.set_xticks(range(X.shape[1]))
ax.set_xticklabels([iris.feature_names[i] for i in indices], rotation=45)
ax.set_xlim([-1, X.shape[1]])
ax.set_title('Feature Importance')
plt.tight_layout()
st.pyplot(fig)
