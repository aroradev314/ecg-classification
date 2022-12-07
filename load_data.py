import pandas as pd
import ast
import funcs
import numpy as np
import pickle

path = "ptbxl/"
data = pd.read_csv(path + 'ptbxl_database.csv')

print(1)
def _eval(x):
    return ast.literal_eval(x)


data.scp_codes = data.scp_codes.apply(_eval)
statements = pd.read_csv(path + 'scp_statements.csv')
print(2)


def aggregate_diagnostic(y_dic):
    tmp = ""
    for key in y_dic.keys():
        if key == "SR" or key == "AFIB":
            tmp = key
            break
    return tmp


data['rhythm'] = data.scp_codes.apply(aggregate_diagnostic)

data = data[data.rhythm != ""]

rate = 100
records = funcs.load_raw_data(data, rate, path)
print(3)


second = []  # stores the readings for the second lead as second lead is best for diagnosis
current = []
for i in records:
    current = []
    for j in range(1000):
        current.append(i[j][1])  # gets second lead from each sample
    second.append(current)
second = np.asarray(second)

diagnostic = np.array(list(map(lambda x: 1 if x == "AFIB" else 0, data['rhythm'].tolist())))


d = {'readings': second.tolist(), 'diagnostic': diagnostic.tolist()}
df = pd.DataFrame(data=d)

train, validate, test = np.split(df.sample(frac=1, random_state=42), [int(.6*len(df)), int(.8*len(df))])

train_readings, train_diagnostic = \
    funcs.to_numpy(train['readings'].to_numpy()), funcs.to_numpy(train['diagnostic'].to_numpy())
validate_readings, validate_diagnostic = \
    funcs.to_numpy(validate['readings'].to_numpy()), funcs.to_numpy(validate['diagnostic'].to_numpy())
test_readings, test_diagnostic = \
    funcs.to_numpy(test['readings'].to_numpy()), funcs.to_numpy(test['diagnostic'].to_numpy())

funcs.pickler(train_readings, 'pickled/train_readings.pkl')
funcs.pickler(train_diagnostic, 'pickled/train_diagnostic.pkl')
funcs.pickler(validate_readings, 'pickled/validate_readings.pkl')
funcs.pickler(validate_diagnostic, 'pickled/validate_diagnostic.pkl')
funcs.pickler(test_readings, 'pickled/test_readings.pkl')
funcs.pickler(test_diagnostic, 'pickled/test_diagnostic.pkl')


