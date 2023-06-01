import pickle
import numpy  as np
import pandas as pd

class ChurnInd(object):
	def __init__(self):
		self.home_path             = '/home/felipe/repos/churn_indi/'
		self.AgeScaler             = pickle.load(open(self.home_path + '/src/parameters/AgeScaler.pkl', 'rb'))
		self.AvgTicketScaler       = pickle.load(open(self.home_path + '/src/parameters/AvgTicketScaler.pkl', 'rb'))
		self.BalanceScaler         = pickle.load(open(self.home_path + '/src/parameters/BalanceScaler.pkl', 'rb'))
		self.CreditScoreScaler     = pickle.load(open(self.home_path + '/src/parameters/CreditScoreScaler.pkl', 'rb'))
		self.EstimatedSalaryScaler = pickle.load(open(self.home_path + '/src/parameters/EstimatedSalaryScaler.pkl', 'rb'))
		self.NumOfProductsScaler   = pickle.load(open(self.home_path + '/src/parameters/NumOfProductsScaler.pkl', 'rb'))
		self.TenureScaler          = pickle.load(open(self.home_path + '/src/parameters/TenureScaler.pkl', 'rb'))
	
	def feature_engineering(self, df2):
		# average ticket
		df2['AvgTicket'] = [np.round(x['Balance'] / x['NumOfProducts'], 2) if x['NumOfProducts'] != 0 else 0 for _, x in df2.iterrows()]
		# drop columns
		df2.drop(['RowNumber', 'Surname', 'CustomerId'], axis=1, inplace=True)
		return df2
	
	def data_preparation(self, df3):
		# credit score
		df3['CreditScore'] = self.CreditScoreScaler.transform(df3[['CreditScore']].values)
		# Age
		df3['Age'] = self.AgeScaler.transform(df3[['Age']].values)
		# Tenure
		df3['Tenure'] = self.TenureScaler.transform(df3[['Tenure']].values)
		# Balance
		df3['Balance'] = self.BalanceScaler.transform(df3[['Balance']].values)
		# Number of Products
		df3['NumOfProducts'] = self.NumOfProductsScaler.transform(df3[['NumOfProducts']].values)
		# Estimated Salary
		df3['EstimatedSalary'] = self.EstimatedSalaryScaler.transform(df3[['EstimatedSalary']].values)
		# Average Ticket
		df3['AvgTicket'] = self.AvgTicketScaler.transform(df3[['AvgTicket']].values)
		# geography - One Hot Encoding
		df3 = pd.get_dummies(df3, prefix='Geography', columns=['Geography'], dtype='int64')
		# gender- One Hot Encoding
		df3 = pd.get_dummies(df3, columns=['Gender'], dtype='int64')
		return df3
	
	def get_prediction(self, model, original_data, test_data):
		# model prediction
		pred = model.predict(test_data)
		# join prediction into original data
		original_data['Exited'] = pred
		return original_data.to_json(orient='records', date_format='iso')
