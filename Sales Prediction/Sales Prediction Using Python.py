# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from urllib.parse import quote
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import sweetviz as sv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



# Load the dataset and perform EDA
sales = pd.read_csv(r"C:\Users\kashinath konade\Desktop\CodSoft Intern for DS\car_purchasing.csv", encoding='unicode_escape')

# Save the data to an SQL database
user = 'root'
pw = 'kashinath@123'
db = 'Assignment_ml'
engine=create_engine(f'mysql+pymysql://{user}:%s@localhost:3306/{db}' % quote(f'{pw}'))
sales.to_sql('Sales_Prediction'.lower(), con = engine, if_exists = 'replace', index = False)


# Read the data from the MySQL database
sql = 'SELECT * FROM Sales_Prediction'
data = pd.read_sql_query(sql, con=engine)

data.info()

data.describe()

data.shape


# Checking for unique values
data["customer e-mail"].unique()

data["customer e-mail"].value_counts()

# Checking Null values
data.isnull().sum()*100/data.shape[0]
# There are no NULL values in the dataset, hence it is clean.


# Data Preprocessing
car = data.drop(["customer name", "customer e-mail", "country"], axis=1)
X = car.drop(["car purchase amount"], axis=1)
Y = car[["car purchase amount"]]


# The above are manual approach to perform Exploratory Data Analysis (EDA). The alternate approach is to Automate the EDA process using Python libraries.
# 
# Auto EDA libraries:
# - Sweetviz
# - dtale
# - pandas profiling
# - autoviz

# 
# # **Automating EDA with Sweetviz:**
# 

# Using sweetviz to automate EDA is pretty simple and straight forward. 3 simple steps will provide a detailed report in html page.
# 
# step 1. Install sweetviz package using pip.
# - !pip install sweetviz
# 
# step2. import sweetviz package and call analyze function on the dataframe.
# 
# step3. Display the report on a html page created in the working directory with show_html function.
# 

# Analyzing the dataset
report = sv.analyze(car)

# Display the report
# report.show_notebook()  # integrated report in notebook

report.show_html('EDAreport.html') # html report generated in working directory


# Scale the data using Min-Max scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
Y_scaled = scaler.fit_transform(Y)



# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(X_scaled, Y_scaled, test_size=0.25, random_state=101)


# Build and train the ANN model
model = Sequential()
model.add(Dense(25, input_dim=5, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(1, activation='linear'))



model.compile(optimizer='adam', loss='mean_squared_error')
epochs_hist = model.fit(xtrain, ytrain, epochs=100, batch_size=50, verbose=1, validation_split=0.2)

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(epochs_hist.history["loss"], label='Training Loss')
plt.plot(epochs_hist.history["val_loss"], label='Validation Loss')
plt.title('Loss During Training or Validation')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()


# Predict car purchase amount for a random sample
X_random_sample = np.array([[1, 55, 65000, 11600, 562341]])
y_predict = model.predict(X_random_sample)
print('Predicted Purchase Amount is =', y_predict[0][0])


""" Conclusion:
 the provided code is a foundational step for a sales prediction. 
 It loads data, builds a machine learning model, and makes sales predictions. 
 However, it's important to note that creating an effective sales prediction system 
 requires additional steps, including data analysis, feature engineering, model evaluation, 
 and the use of real-world data. The code is a starting point but not a complete solution for sales prediction."""