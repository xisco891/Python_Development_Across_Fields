
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv("campaign_budget.csv")
final_report = pd.read_csv("final_report.csv", sep=";")

##Cleaning Data
dataset = dataset.drop('Unnamed: 0', axis=1)
final_report = final_report.drop(['Unnamed: 0',
                                  'date',
                                  'postview_conversion_ZorbyMax_desktop_startseite',
                                  'postclick_conversion_ZorbyMax_desktop_startseite'],
                                  axis=1)


##Exploring the Categorical Data -> Encoding of the Categorical Data

cat_data_report = final_report.select_dtypes(include=['object']).copy()
non_cat_data = final_report.select_dtypes(exclude=['object']).copy()
##There seems to be categorical_data that looks nummerical-> Fixing the error. 
cat_data_report.iloc[:,3] = pd.to_numeric(cat_data_report.iloc[:,3], errors='coerce')
cat_data_report.iloc[:,4] = pd.to_numeric(cat_data_report.iloc[:,4], errors='coerce')
cat_data_report.iloc[:,5] = pd.to_numeric(cat_data_report.iloc[:,5], errors='coerce')
cat_data_report.iloc[:,6] = pd.to_numeric(cat_data_report.iloc[:,6], errors='coerce')
cat_data_report.iloc[:,7] = pd.to_numeric(cat_data_report.iloc[:,7], errors='coerce')
cat_data_report.iloc[:,8] = pd.to_numeric(cat_data_report.iloc[:,8], errors='coerce')

#####################################################################################


non_cat_data = pd.concat([non_cat_data, cat_data_report.iloc[:, 3:]],axis=1)
cat_data_report = cat_data_report.iloc[:, 0:3]

########################################################################################
####Adding Budget Column from 'Campaign_Budget' to 'final_report' Dataset##############


X_campaign = dataset.iloc[:,1:].values
n = X_campaign.shape[0]
budget_dict = {X_campaign[i,0] : X_campaign[i,1] for i in range(0,n)}
vector_budget = np.zeros((cat_data_report.values.shape[0],1), dtype=float)

#Filling new column with budget values from campaign budget data. 
n_samples = non_cat_data.shape[0]
n_var = non_cat_data.shape[1]

for i in range(0, n_samples):
    index = cat_data_report.iloc[i,0]
    vector_budget[i] = budget_dict[index]   

#######One_Hot Encoding for Categorical Data from Final_Report
X_onehot = cat_data_report.copy()
encoded_X = pd.get_dummies(X_onehot, columns=['campaign_name'], prefix=['C_Name_'])
encoded_X = pd.get_dummies(encoded_X, columns=['contentunit_name'], prefix=['content_unit'])
encoded_X = pd.get_dummies(encoded_X, columns=['banner_name'], prefix=['banner_name'])


####Prepare the Independent and Dependent Variables
y = non_cat_data.iloc[:, 0].values
non_cat_data = non_cat_data.drop(['adrequests'], axis=1)
X = pd.concat([encoded_X, non_cat_data],axis=1).values
X = np.append(X,vector_budget, axis=1)


###Imputing values for Missing Data in X
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X)

###Imputing values for Missing Data in Y
from sklearn.preprocessing import Imputer
imputer_y = Imputer(missing_values='NaN', strategy = 'mean', axis = 0)
y = y.reshape(-1,1)
imputer_y = imputer_y.fit(y)
y = imputer_y.transform(y)

#
## Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300,random_state=0)
regressor.fit(X,y)
              
# Predicting a new result
y_pred = regressor.predict(X)

# Visualising the Regression results (for higher resolution and smoother 
#curve)

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()