import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
file = "C:/Pyth/Intro to BigData/한국_기업문화_HR_데이터셋_샘플.csv"
df = pd.read_csv(file)
print(df.head())

new_columns = ['근무환경만족도', '업무만족도', '워라밸','야근여부', '직급', '출장빈도', '연봉인상률', '총경력', '이전회사경험수','현회사근속년수']
df1 = df[new_columns].copy()  

target = df['이직여부'].apply(lambda x: 0 if x == "Yes" else 1)
target.name = '이직여부 이진화'

df.drop(columns=['이직여부'], inplace=True)

df = pd.concat([target, df1], axis=1)

print(df.columns)
print(df.head(5))

#결측치 처리 - 존재 (X)
print(df.isnull().sum())
print(df.info())
print(len(df)) # = every column has 1000 rows

df['야근여부'] = df['야근여부'].apply(lambda x: 0 if x == "Yes" else 1)
print(df.columns)
print(df.head(5))

print(df.info())
# 범주형 변수 인코딩
cat_cols = df.select_dtypes(include=['object', 'category']).columns
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col])

print(df1.head(5))



#데이터에 1000개의 행이 있음을 위에서 확인
raw = df
np_raw = raw.values

train = np_raw[:800]
test = np_raw[800:]

y_train = [i[0] for i in train] #이직여부 = col[0]
X_train = [j[1:] for j in train] 
y_test = [i[0] for i in test]
X_test = [j[1:] for j in test]
print(len(X_train), len(y_train), len(y_test), len(X_test))

print(df.info())


y = raw['이직여부 이진화']
X = raw.drop(['이직여부 이진화'], axis=1)

#StandardScaler
ss = StandardScaler()
X_ss = ss.fit_transform(X)
X_ss_pd = pd.DataFrame(X_ss, columns=X.columns)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Standardized Logistic Regression
X_out = X_ss_pd
X_train, X_test, y_train, y_test = train_test_split(X_out, y, test_size=0.2, random_state=13)
log_reg = LogisticRegression(random_state=13, solver='liblinear', C=10.0)
log_reg.fit(X_train, y_train)
pred = log_reg.predict(X_test)
print("StandardScaler Accuracy:", accuracy_score(y_test, pred))
print("StandardScaler Confusion Matrix:\n", confusion_matrix(y_test, pred))
#정확도와 혼동 행렬을 보아 모델이 이직한 사람을 잘 예측하고 있음을 알 수 있음
#하지만 실제로 이직하였으나 예측하지 못한 경우가 꽤 있음 ~14%

print("Logistic Regression Coefficients: ", log_reg.coef_)
print((df['이직여부 이진화'] == 0).sum(), "이직한 사람의 수")

prob = log_reg.predict_proba(X_test)
print(prob[:5])
top5 = sorted(prob, key = lambda x: x[0], reverse=True)[:5]
print("Top5:", top5)

# ====== 신규 데이터 New_A, New_B, New_C ======

New_A = {
    "근무환경만족도": 2, "업무만족도": 2, "워라밸": 2, "야근여부": "Yes", "직급": "사원", "출장빈도": "Travel_Rarely",
    "연봉인상률": 12, "총경력": 4, "이전회사경험수": 1, "현회사근속년수": 1
}

New_B = {
    "근무환경만족도": 3, "업무만족도": 4, "워라밸": 3, "야근여부": "No", "직급": "대리", "출장빈도": "Non-Travel",
    "연봉인상률": 14, "총경력": 18, "이전회사경험수": 2, "현회사근속년수": 7
}

New_C = {
    "근무환경만족도": 1, "업무만족도": 1, "워라밸": 2, "야근여부": "Yes", "직급": "과장", "출장빈도": "Travel_Frequently",
    "연봉인상률": 11, "총경력": 10, "이전회사경험수": 3, "현회사근속년수": 2
}

new_df = pd.DataFrame([New_A, New_B, New_C])

# '야근여부'도 0/1 변환
new_df['야근여부'] = new_df['야근여부'].apply(lambda x: 0 if x == "Yes" else 1)

# 범주형 변수 인코딩 (기존 학습 때 사용한 LabelEncoder 재사용)
cat_cols = new_df.select_dtypes(include=['object', 'category']).columns
for col in cat_cols:
    new_df[col] = LabelEncoder().fit_transform(new_df[col])

# 스케일링
new_scaled = scaler.transform(new_df)

# 예측 (이직 확률)
probs = log_reg.predict_proba(new_scaled)[:, 0]  # 0: 이직 'Yes' 클래스 확률

labels = ['A', 'B', 'C']
for i, (label, prob) in enumerate(zip(labels, probs), start=1):
    print(f"New_{label} 이직 가능성 (0=이직, 1=잔류) 확률:", prob)


