Our code consists of 7 main parts:

1. Regressor class
    a. Constructor
        The constructor in addition to the original values for this function takes number of epochs, learning rate, neurons, and decay as input parameters. neurons is an array [x, y], where x is the number of output neurons of the input layer, and y is the number of output neurons of the hidden layer. The input parameters do have default values, so do not have to be explicitly set when the constructor is called.
        The rest of this class remains the same as the skeleton code.

    b. Score
        The score function computes the RMSE of input data x, y

2. RegressorHyperParameterSearch
    This function takes in a regressor object, x and y validation and training datasets, a boolean, and the number of combinations (set to 20 by default) which is used for the random search. 
    This function will call the grid search function if the bool is True, or the random search function if the bool False.

    This function returns the hyperparameters found by grid search or random search.

3. Grid search
    This function takes a regressor object and x and y validation and training datasets.
    It returns the optimal number of epochs and learning rate for the model that resulted in the lowest error on the validation set. 

4. Random search
    This function takes a regressor object, x and y validation and training datasets, and number of combinations.
    It returns the optimal number of epochs, learning rate, number of neurons (in the same [x, y] format as input/output of the hidden layer), and weight decay for the model that resulted in the lowest error on the validation set.
    It also will save the pickle file of the optimal model.

5. Create test data
    This function takes x and y training and testing and a filename.
    It will create and evaluate 50 regressor objects export the data into filename.csv

6. eval_test data
    This function takes in a filename (.csv).
    It will print statistical information related to this (mean, standard deviation etc)

7. main
    This function has no arguments.
    In this function we read the data, and split it into training/testing/validation.
    This function will create and save a pickle file of a regressor object.
    It will also run hyperparameter search and print the hyperparameter values found.
