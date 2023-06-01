import torch
import pickle
import math
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
from torch import nn
from torch.optim import Adam
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

import random
import itertools

class Regressor(nn.Module):

    def __init__(self, x, nb_epoch=1000, learning_rate=0.01, neurons=[64, 32], decay=0):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """
        Initialise the model.

        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape
                (batch_size, input_size), used to compute the size
                of the network.
            - nb_epoch {int} -- number of epoch to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own <- using this as it probably useful
        X, _ = self._preprocessor(x, training=True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch
        self.scaler = None
        self.lr = learning_rate
        self.encoder = None
        self.y_scale = None
        self.decay = decay

        #inherit from torch.nn module, call constructor from superclass
        super().__init__()

        #The linear function automatically sets the weights and bias to uniform dist

        self.layers = nn.Sequential(
            #input layer
            nn.Linear(self.input_size, neurons[0]),
            nn.ReLU(),
            #hidden layer
            nn.Linear(neurons[0], neurons[1]),
            nn.ReLU(),
            #output layer
            nn.Linear(neurons[1], self.output_size),
        )

    def forward(self, x):
        #we have regression so using linear func
        out = self.layers(x)
        return out
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def _preprocessor(self, x, y = None, training = False):
        """
        Preprocess input of the network.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
            - training {boolean} -- Boolean indicating if we are training or
                testing the model.

        Returns:
            - {torch.tensor} -- Preprocessed input array of size
                (batch_size, input_size).
            - {torch.tensor} -- Preprocessed target array of size
                (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        # Return preprocessed x and y, return None for y if it was None

        #One hot encoding
        if training is True:
            self.encoder = ColumnTransformer([('_', OneHotEncoder(handle_unknown='ignore'), ["ocean_proximity"])], remainder='passthrough')
            self.encoder.fit(x)

        transform = self.encoder.transform(x)
        x = pd.DataFrame(transform)

        #Dealing with NaN values
        for column in x.columns:
            average_val = x[column].mean()
            x[column].fillna(average_val, inplace=True)

        #Scaling the input
        x_values = x.values #returns a numpy array
        if training is True:
            self.scaler = preprocessing.MinMaxScaler().fit(x_values)

        x_scaled = self.scaler.transform(x_values)
        #Turning data and output from pd.dataframe into torch.tensor format
        data_tensor = torch.tensor(x_scaled)

        #Same process for y
        if y is not None:
            if training is True:
                self.y_scale = preprocessing.MinMaxScaler().fit(y.values)
            y_scaled = self.y_scale.transform(y.values)
            output_tensor = torch.tensor(y_scaled)
        else:
            output_tensor = None

        return data_tensor, output_tensor

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y=y, training=True) # Do not forget

        #adding optimiser, learning rate and/or optimizer type may need changing (?) just using this for now
        opt = optim.Adam(self.parameters(), self.lr, weight_decay=self.decay)

        #loss func, is there other functions we should use for this?
        crit = nn.MSELoss()

        #We want to keep track whether accuracy is increasing or not
        best_acc = 0.0

        #loop trainer nb_epoch times
        for i in range(self.nb_epoch):

            #Reset this epoch's accuracy
            train_acc = 0.0

            #Clear existing gradients
            #Gradients will be accumulated when we do loss.backward() if we do not clear them
            opt.zero_grad()

            #predicted outputs = inputs.forward
            outputs = self(X.float())
            #targets = dataset outputs
            targets = Y.float()

            #find loss
            loss = crit(outputs, targets)

            #backprop
            loss.backward()

            #optimise
            opt.step()

        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


    def predict(self, x):
        """
        Ouput the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).

        Returns:
            {np.darray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training=False) # Do not forget

        #send data into the nn to find the output
        out = self(X.float())
        #invert the scaling
        out = self.y_scale.inverse_transform(out.detach().numpy())

        return out

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw ouput array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        out = self.predict(x)

        #finding RMSE
        diff = pd.DataFrame.sum((pd.DataFrame.abs(out - y)) ** 2)
        err = diff / len(y)

        return math.sqrt(err)

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model):
    """
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor():
    """
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model


def RegressorHyperParameterSearch(regressor, x_validation, y_validation, x_train, y_train, tuning_search=False, n=20):
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.

    Returns:
        The function should return your optimised hyper-parameters.

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    if tuning_search is True:
        print("<---- TUNING USING GRID SEARCH ---->\n")
        return grid_search(regressor, x_validation, y_validation, x_train, y_train)

    return random_search(regressor, x_validation, y_validation, x_train, y_train, n) # Return the chosen hyper parameters


def grid_search(regressor, x_validation, y_validation, x_train, y_train):
    smallest_error = math.inf
    best_epoch = 0.0
    best_lr = 0.0
    #initialise
    regressor.lr = 0.1
    regressor.nb_epoch = 10
    start = time.perf_counter()
    for n in range(20):
        regressor.nb_epoch = 10

        for m in range(100):
            regressor.nb_epoch += 10
            regressor = Regressor(x_train, regressor.nb_epoch, regressor.lr)
            regressor.fit(x_train, y_train)
            X,Y = x_validation, y_validation
            # Error
            error = regressor.score(X, Y)
            '''print("Current ERROR:", error)
            print("Current EPOCH: ", regressor.nb_epoch)
            print("Current LR: ", regressor.lr)
            print("<-------->")'''
            if(error < smallest_error):
                best_epoch = regressor.nb_epoch
                best_lr = regressor.lr
                smallest_error = error

        regressor.lr -= 0.005

    end = time.perf_counter()
    diff = end - start
    print("best EPOCH: ", best_epoch)
    print("best LR: ", best_lr)
    print("best ERROR: ", smallest_error)
    print("time taken: ", diff)
    print("\n")

    return best_epoch, best_lr, [], 0

def random_search(regressor, x_validation, y_validation, x_train, y_train, n):
    smallest_error = math.inf
    best_epoch = 0.0
    best_lr = 0.0
    best_neurons = []
    best_decay = 0.0
    #initialise
    regressor.lr = 0.1
    regressor.nb_epoch = 10
    start = time.perf_counter()
    #list all epochs/lrs you wish here
    epochs = [400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
    lrs = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2]
    neuron_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    decays = [0, 0.000001, 0.00001, 0.0001]
    #create a random sample of data in the form [(x, y), ...(xn, yn)] where x and y are the datasets to chose from an n is the number of combinations to be made
    #now create random samples of all variables
    tests = random.sample(set(itertools.product(epochs, lrs, neuron_values, neuron_values, decays)), n)

    tests = np.array(list(tests))

    for i in tests:
        #extract the data values from the random sample
        nb_epoch = int(i[0])
        lr = i[1]
        neurons = [i[2].astype(int), i[3].astype(int)]
        decay = i[4]

        #create the model and find error
        regressor = Regressor(x_train, nb_epoch, lr, neurons, decay)
        regressor.fit(x_train, y_train)
        error = regressor.score(x_validation, y_validation)

        #compare the error to the current best, if error is improved replace current best values
        if error < smallest_error:
            #save the current best model
            save_regressor(regressor)
            #update the best hyperparameters
            smallest_error = error
            best_epoch = nb_epoch
            best_lr = lr
            best_neurons = neurons
            best_decay = decay

    end = time.perf_counter()
    diff = end - start
    print("best ERROR: ", smallest_error)
    print("best EPOCH: ", best_epoch)
    print("best LR: ", best_lr)
    print("best NEURONS: ", best_neurons)
    print("best DECAY: ", best_decay)
    print("time taken: ", diff)
    print("\n")

    return best_epoch, best_lr, best_neurons, best_decay

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################

#These two functions are for testing how target scaling affects RMSE, and any other interesting findings in future
def create_test_data(x_train, y_train, x_testing, y_testing, filename):
    err_arr = []

    for i in range(50):
        regressor = Regressor(x_train)
        regressor.fit(x_train, y_train)
        error = regressor.score(x_testing, y_testing)
        err_arr.append(error)
        print(i, error)
    data = pd.DataFrame(err_arr)
    data.to_csv (filename, index = False, header=True)

def eval_test_data(filename):
    data = pd.read_csv(filename)
    print(data.describe())


def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas Dataframe as inputs
    data = pd.read_csv("housing.csv")

    # Spliting input and output
    decile = int(0.1*len(data))
    x_all = data.loc[:, data.columns != output_label]
    y_all = data.loc[:, [output_label]]
    x_train = x_all.iloc[0:(8*decile), :]
    x_validation = x_all.iloc[(8*decile):(9*decile), :]
    x_testing = x_all.iloc[(9*decile):, :]

    y_train = y_all.iloc[0:(8*decile), :]
    y_validation = y_all.iloc[(8*decile):(9*decile), :]
    y_testing = y_all.iloc[(9*decile):, :]

    # Training
    # This example trains on the whole available dataset.
    # You probably want to separate some held-out data
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train)
    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    #If false, run random_search, if true, run grid_search
    search_type = False
    combinations = 30
    best_epoch, best_lr, best_neurons, best_decay = RegressorHyperParameterSearch(regressor, x_validation, y_validation, x_train, y_train, search_type, combinations)

    print("\n<---- FINAL RESULTS ---->")
    print("best EPOCH: ", best_epoch)
    print("best LR: ", best_lr)
    print("best NEURONS: ", best_neurons)
    print("best DECAY: ", best_decay)


if __name__ == "__main__":
    example_main()
