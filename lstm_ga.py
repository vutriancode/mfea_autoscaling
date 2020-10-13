import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from math import sqrt, ceil, floor, log2
import itertools, random


def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        v = dataset[i] - dataset[i-interval]
        diff.append(v)
    return np.array(diff)


def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


# X (sample, features), y numpy array
# return X: (samples, timesteps, features)
#        y: expected values
def create_dataset(X, y, time_steps=1):
    features = X.shape[1]
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i : i+time_steps]
        Xs.append(v)
        ys.append(y[i + time_steps])
    Xs, ys = np.array(Xs), np.array(ys)
    return Xs, ys


# dataset 2D np array
# return scaler, scaled values
def scale(dataset):
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler = scaler.fit(dataset)
    scaled_dataset = scaler.transform(dataset)
    return scaler, scaled_dataset


def invert_scale(scaler, data):
    return scaler.invert_scale(data)


def split(X, y, train_size=0.8):
    n_train = int(len(X) * train_size)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    return X_train, y_train, X_test, y_test

def build_model(input_shape, lstm_units=128):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(units=lstm_units, input_shape=input_shape))
    model.add(keras.layers.Dense(units=32))
    model.add(keras.layers.Dense(units=1)) 
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    # keras.utils.plot_model(model, show_shapes=True)
    # plt.show()
    return model

def train_column(col, batch_size=32, time_steps=20, epochs=10):
    raw_values = df.iloc[:,col].values
    scaler, scaled_values = scale(raw_values.reshape(raw_values.shape[0], 1))

    X, y = create_dataset(scaled_values, scaled_values, time_steps=time_steps)
    X_train, y_train, X_test, y_test = split(X, y, 0.8)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    model = build_model(X_train[0].shape, lstm_units=128)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=0,
        shuffle=False)
    
    plt.plot(history.history['loss'])
    plt.show()

    y_pred = model.predict(X_test)
    loss = keras.losses.MeanSquaredError()
    print('y_test shape: ', y_test.shape)
    print('rmse loss in test set: before inverse scale = %f' %(sqrt(loss(y_pred, y_test))))

    # plt.figure(figsize=(25, 5))
    y_test_inverse = scaler.inverse_transform(y_test)
    y_pred_inverse = scaler.inverse_transform(y_pred)

    print('rmse loss  test set: after inverse scale = %f' %(sqrt(loss(y_pred_inverse, y_test_inverse))))
    plt.plot(y_test_inverse, color='green', linewidth=0.5)
    plt.plot(y_pred_inverse, color='red', linewidth=0.5)
    plt.legend(['y_true', 'y_pred'])
    plt.show()
    return sqrt(loss(y_pred_inverse, y_test_inverse))


df = pd.read_csv('./data/input_data/google_trace/1_job/3_mins.csv',header = None)


# train_column(col=1, batch_size=32, time_steps=20, epochs=5)

class Individual:
    def __init__(self, genes):
        self.genes = genes
        self.fitness = train_column(col = 3, 
                            batch_size=self.genes[0], 
                            time_steps=self.genes[1], 
                            epochs=3)
    def __str__(self):
        return "(Genes: {0}, Fitness: {1})".format(self.genes, self.fitness)
        
class Population:
    def __init__(self, individuals):
        self.individuals = individuals
    
    def getAllFitness(self):
        return [ind.fitness for ind in self.individuals]
    
    def sortIndividual(self):
        allFitness = self.getAllFitness()
        self.individuals.sort(key=lambda ind: ind.fitness)
        
    def fitSplit(self, num):
        self.sortIndividual()
        return self.individuals[:num], self.individuals[num:]    
    
    def __str__(self):
        return str([str(individual) for individual in self.individuals])
    
class SimpleGA:
    def __init__(self, population):
        self.population = population
    
    @classmethod
    def crossOver(cls, ind1, ind2):
        print('Cross over individual ', ind1, " and ", ind2)
        newIndGenes = [(ind1.genes[i] + ind2.genes[i]) // 2 for i in range(len(ind1.genes))]
        return Individual(newIndGenes)
    
    @classmethod
    def select(cls, size):
        g = itertools.combinations(range(size), 2)
        pairList = list(g)
        random.shuffle(pairList)
        return pairList[:size]
    
    @classmethod
    def mutate(cls, ind):
        if random.random() < 0.5:
            ind.genes[0] = 2 ** ceil(log2(ind.genes[0]))
        return ind    
    
    def GA(self):
        pairList = SimpleGA.select(len(population.individuals) // 2)
        fit, unfit = self.population.fitSplit(len(population.individuals) // 2)
        print("Fit population: ", [str(individual) for individual in fit])
        print("Unfit population ", [str(individual) for individual in unfit])
        newPopulation = []
        for ind in fit:
            newPopulation.append(ind)
        for ind1, ind2 in pairList:
            newPopulation.append(SimpleGA.crossOver(fit[ind1], 
                                                    fit[ind2]))
        for ind in unfit:
            ind = SimpleGA.mutate(ind)     
            newPopulation.append(ind)
        
        nextGen = Population(newPopulation)
        nextGen.sortIndividual()
        self.population = Population(nextGen.individuals[:len(population.individuals)])
    
    def __str__(self):
        return str(self.population)

individuals = [[100, 16], [80, 8], [64, 16], [64, 32], [50, 12], [32, 20], [32, 8], [16, 10], [8, 10]]        
population = Population([Individual(ind) for ind in individuals])
[print(individual) for individual in population.individuals]
print(population)
myGA = SimpleGA(population)

result = []
for i in range(4):
    myGA.GA()
    result.append(str(myGA.population))
    
print("Begin")
print(population)
for i in range(4):
    print("Generation ", i+1)
    print(result[i])
        