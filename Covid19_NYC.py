import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from tensorflow.keras.layers.experimental import preprocessing

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import dill


def get_data():
    
    data = pd.read_csv('data-by-day.csv')

    numeric_date = np.linspace(0,len(data)-1,len(data))

    data = data.drop(columns=['date_of_interest'])

    data['date'] = numeric_date

    Y = data['CASE_COUNT']
    #X = data.drop(columns=['CASE_COUNT','INCOMPLETE','CASE_COUNT_7DAY_AVG','ALL_CASE_COUNT_7DAY_AVG'])
    X = data[['date', 'HOSPITALIZED_COUNT', 'DEATH_COUNT', 'DEATH_COUNT_7DAY_AVG']]
    Y = Y.values
    X = X.values
    return X, Y

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(58, input_dim=58, kernel_initializer='normal', activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mae', optimizer='adam')
    return model

def ConvNN_model():
    model = Sequential()
    model.add(Conv1D(3, 2, activation="relu", input_shape=(4,1)))
    model.add(Flatten())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1))
    model.compile(loss="mae", optimizer="adam")
    return model

if __name__ == '__main__':
    X, Y = get_data()
    xtrain, xtest, ytrain, ytest = train_test_split(X,Y, test_size=0.2)

    """model = baseline_model()
    history = model.fit(xtrain, ytrain, epochs=150, batch_size=12,  verbose=0, validation_split=0.1)

    pred = model.predict(xtest)
    print(model.evaluate(xtrain,ytrain))
    print("MSE: %.4f" % mean_squared_error(ytest, pred))
    print("MAE: %.4f" % mean_absolute_error(ytest, pred))

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig('mae_regression_baseline.png')
    plt.show()

    x_axis = range(len(pred))
    plt.scatter(x_axis, ytest, s=5, color='blue')
    plt.plot(x_axis, pred, color='red', lw=0.8)
    plt.ylabel('Num. of Cases')
    plt.xlabel('Test Number')
    plt.legend(['Actual','Predicated'])
    plt.savefig('test_baseline.png')    
    plt.show()"""
    
    xtrain = np.expand_dims(xtrain, axis = -1)
    xtest = np.expand_dims(xtest, axis = -1)

    model = ConvNN_model()
    model.summary()
    history = model.fit(xtrain, ytrain, batch_size=12,epochs=200, verbose=0, validation_split = 0.1)
    pred = model.predict(xtest)
    print(model.evaluate(xtrain, ytrain))
    print("MSE: %.4f" % mean_squared_error(ytest, pred))
    print("MSE: %.4f" % mean_absolute_error(ytest, pred))
    model.save('model')
    model = keras.models.load_model('model')
    print(model.predict([[[450], [20000], [1000], [10000]]]))
    
    
    """dill.dump(model, open('model.pkl','wb'))    
    # Loading model to compare the results
    model = dill.load(open('model.pkl','rb'))
    # print(model.predict([[2, 9, 6]]))"""

    """x_axis = range(len(pred))
    plt.scatter(x_axis, ytest, s=5, color='blue')
    plt.plot(x_axis, pred, lw=0.8, color='red')
    plt.legend(['Actual','Predicated'])
    plt.savefig('test_conv.png')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train','Validation'], loc='upper right')
    plt.savefig('mae_regression_conv.png')
    plt.show()"""
    
    
    
    
    
