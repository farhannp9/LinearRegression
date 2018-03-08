import pandas as pd
import seaborn as sb
import numpy as np
from sklearn import linear_model

sb.set()

#working environment is different, delete this if same
current_file = 'C:\Users\user\git\\ristek\\'

training_data = pd.read_csv(current_file + 'train.csv', index_col = 0)

print("number of training data : " +str(len(training_data)))

dfs = np.split(training_data, [1], axis=1)
X = dfs[0].as_matrix()
y = dfs[1].transpose().as_matrix()[0]
#X = dfs[0].as_matrix()
#y = dfs[1].as_matrix()

print(X[:3])
print(y[:3])

lr = linear_model.LinearRegression()

lr.fit(X,y)


test_data = pd.read_csv(current_file + 'test.csv', index_col = 0)

#print(test_data[:3])

res = lr.predict(test_data)
resu = [res]
resu = np.transpose(resu)

print(resu)


resu = pd.DataFrame(resu, index = test_data.index.copy())

resu.columns = ['Brain']

submission = resu
print(submission)
submission.to_csv(path_or_buf = current_file + 'subs.csv')
#print(submission[:3])
#print (len(resu))
#print(resu[:31])
#print(resu[31:])
