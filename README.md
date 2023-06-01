# Regression Neural Network

This project aims to implement a neural network architecture for regression using the California House Prices Dataset. The goal is to predict the median house value of a block group based on various attributes.

## Dataset
The California House Prices Dataset consists of 20,640 observations on ten variables:

1. longitude: longitude of the block group
2. latitude: latitude of the block group
3. housing median age: median age of the individuals living in the block group
4. total rooms: total number of rooms in the block group
5. total bedrooms: total number of bedrooms in the block group
6. population: total population of the block group
7. households: number of households in the block group
8. median income: median income of the households in the block group
9. ocean proximity: proximity to the ocean of the block group
10. median house value: median value of the houses in the block group

## Implementation
This GitHub repository consists of code that can be divided into 7 main parts:

1. Regressor class: The Regressor class includes a constructor that initializes the necessary values for the model, such as the number of epochs, learning rate, number of neurons, and decay. It also contains a score function to compute the root mean square error (RMSE) of the input data.

2. RegressorHyperParameterSearch: This function performs hyperparameter search using a regressor object, validation and training datasets, and a boolean flag to determine the type of search (grid search or random search). It returns the optimal hyperparameters found.

3. Grid search: This function conducts a grid search to find the optimal number of epochs and learning rate for the model based on the validation set's performance.

4. Random search: This function performs a random search to find the optimal hyperparameters, including the number of epochs, learning rate, number of neurons in the hidden layer, and weight decay. It saves the optimal model using pickle.

5. Create test data: This function generates and evaluates 50 regressor objects and exports the data into a CSV file.

6. eval_test data: This function takes a CSV filename as input and prints statistical information related to the data, such as the mean and standard deviation.

7. main: The main function reads the data, splits it into training, testing, and validation sets. It creates and saves a pickle file of a regressor object. Additionally, it performs a hyperparameter search and prints the found hyperparameter values.

## Contributing
Contributions to this project are welcome! If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with descriptive messages.
4. Push your changes to your forked repository.
5. Submit a pull request, explaining the changes you have made.
6. Please ensure that your contributions align with the project's coding style and guidelines.

## License
This project is licensed under the MIT License. You are free to use, modify, and distribute the code in this repository for personal or commercial purposes. However, please note that any contributions you make to this project will be subject to the same license terms.

