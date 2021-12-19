
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

option = st.selectbox(
'Model Selection',
('Random Forest Regressor', 'Linear Regression'))

st.write('You selected:', option)

st.title("Model Deployment: Random Forest Regressor")
st.sidebar.header("User Input Parameters")

def user_input():
    runs = st.sidebar.number_input("Enter Runs")
    wickets = st.sidebar.number_input("Enter Wickets")
    overs = st.sidebar.number_input("Enter overs")
    stricker = st.sidebar.number_input("Enter Strickers")
    nonstricker = st.sidebar.number_input("Enter Non - Strickers")

    data = {'runs': runs, 'wickets': wickets, 'overs': overs, 'striker': stricker, ' non-striker': nonstricker}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input()
st.subheader('User Inputs Fetched From Sidebar')
st.write(df)

odi = pd.read_csv(r"C:\Users\Shreyas Vaishnav\Desktop\CricketScorePredictor-master\data\odi.csv")
odi = odi.dropna()

x = odi.iloc[:,[7,8,9,12,13]]
y = odi.iloc[:,14]

ypoints = y #np.array([3, 8, 1, 10])
st.line_chart(ypoints)
#st.sidebar.area_chart(ypoints)
#st.vega_lite_chart(ypoints)
#st.barplot(ypoints)
#st.bar_chart(odi["striker"][100])

#Splitting Dataset into 4 Splits
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#@st.cache(suppress_st_warning=True)
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators=20,max_features=None) # verbose= 3)
reg.fit(X_train,y_train)

from sklearn.linear_model import LinearRegression
#model = LinearRegression()
#model.fit(X_train, y_train)

y_pred = reg.predict(X_test)
#y_pred = model.predict(X_test)


prediction = reg.predict(df)
#prediction = model.predict(df)

st.subheader('Predicted Result')
st.write(prediction)

st.subheader('R Squared Score: ')
st.write(r2_score(y_pred, y_test)*100)
st.subheader('Mean Absoute Error: ')
st.write(mean_absolute_error(y_pred, y_test))
st.subheader('Mean Squared Error: ')
st.write(mean_squared_error(y_pred, y_test))
