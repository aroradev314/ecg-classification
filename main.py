import funcs

path = "pickled/"
trainReadings = funcs.unpickler(path + 'train_readings.pkl')

print(funcs.easy_cwt(trainReadings, [38], 'mexh'))
