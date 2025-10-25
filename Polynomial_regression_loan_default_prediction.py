import pandas
import numpy
import matplotlib.pyplot
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

data = pandas.read_csv('loan_default_rate_dataset.csv')
print(data)

x = data[['Credit_Score','Annual_Income','Loan_Amount']]
y = data['Loan_Default_Probability']

poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)

model = LinearRegression()
model.fit(x_poly, y)

predicted_y = model.predict(x_poly)
print(predicted_y)

pyplot.figure(figsize=(12,8))
pyplot.scatter(range(len(y)),y,color='blue',label='Actual Default')
pyplot.scatter(range(len(y)),predicted_y,color='red',label='Predicted Default')
pyplot.legend()
pyplot.xlabel('Rows')
pyplot.ylabel('Loan Default Rate')
pyplot.show()

r2 = model.score(x_poly,y)
print(r2)