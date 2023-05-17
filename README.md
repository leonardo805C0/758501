# AIRPLANE FARE PREDICTION

Leonardo Bosco, Edoardo D’Onghia, Francesco Mazzilli

### Introduction

The aim of this machine learning project is to develop a regression model that could predict accurately the price of flights. The latter fluctuates a lot and depends on many variables such as class of the ticket, how many stops the airplane does etc.

The dataset consisted of 12 variables and more than 300 thousand observations

  

### 1 - Data Loading, Cleaning and Exploration:

We started our code by importing the most important libraries, such as Pandas, and loading the flight dataset from the corresponding CSV file.

  

We then started with a traditional data cleaning procedure that involved:

-   checking if there were any missing values in each column using the isna().sum() method. The dataset has no null values
    
-   checking for duplicates if there are any
    
-   be aware of errors and outliers
    
-   drop non-meaningful variables: "Unnamed: 0" and “flight”. The first because it was a key for the dataset and the second because was a unique code for each flight
    

  

### 1.2 - Exploratory Data Analysis:

  

In order to gain a deeper understanding of our dataset, we conducted visualizations on both the categorical and numerical variables.

  

For the categorical variables, we started by creating bar plots to display the unique values and their respective frequencies. These bar plots provide a comprehensive overview of the different categories present in the dataset and allow us to observe the relative occurrence of each category. Although we have only included a few examples in this report, the complete set of bar plots can be found in our code.

  

Moving on to the numerical variables, our objective was to examine their distributions more closely. To achieve this, we generated histograms where the x-axis represents the variable values and the y-axis represents the frequency of those values. By analyzing these histograms, we gain insights into the distribution patterns and the range of values for each numerical variable.

The most important is price, which is our response variable. It has a huge 0 spike that we must solve to do linear models such as linear regression. Days left is another strange distribution since it could be also treated as a categorical. But since there are many variables we prefer to maintain it as numerical.


  

Finally, in order to gain deeper insights into the distribution of variables within our dataset, we created boxplots for each categorical variable with respect to each numerical one. This form of visualization enables us to identify the median values and pinpoint any outliers.

  
  



Here we can see the relation between duration and airline and we see that Vistara and Air_India are the one with the most outliers

  


This is another useful graph to see at first glance that price goes up with the business class but thanks to some outliers some economy class che reach the business class prices.

  

### 1.3 - Encoding:

Since we have several categorical variables, we wanted to understand how many unique values each of those had and so we checked it for "airline," "source_city," "stops," "destination_city," and "class."

  

In order to deal with those variables we then transformed them into dummies using the ‘get_dummies’ function. It encodes the columns specified in the ‘cat_vars’ list and drops the first category of each column to avoid the "dummy variable trap".

  

The encoded DataFrame is stored in the ‘df_encoded’ variable. Afterwards we concatenated the original DataFrame df with the encoded one using the concat method in a way to have a single dataframe that contained all the predictors and the response variable to prepare for the next task.

  

### 1.4 - Correlation Matrix:

With the ‘corr’ method, we performed a correlation matrix of the encoded DataFrame to understand if there are any important relationships and which is the variables most correlated with price, our response variable

  

Since there were too many variables and the matrix was too big (we have many dummy variables) we managed to extrapolate only the price line of the correlation matrix.

In general, there are not many outstanding relationships except for the ‘class_Economy’ variable which has -0.94 and thus will become one of the best predictors for our models.

  

The correlation is negative because in this variable we find a 1 when the class is “Economy” while we find a 0 when it is “Business”.

  

Having this insight we know from our own experience that Economy class is the cheaper one so when there is a 1 we expect the price to drop and this gives them an inversely proportional relationship.

  

### 1.5 - Data Splitting & Scaling:

To prepare our dataset to modeling we splitted the encoded DataFrame into training and validation sets using the train_test_split function to assign the training and validation feature matrices to X_train and X_validation, respectively, and the corresponding target variables to y_train and y_val. We decided to not use any kind of stratification since there are not unbalanced classes and our y, being continuous, wasn’t suitable for stratification.

  

After this we can scale the numerical columns "duration" and "days_left" to prepare them as well for modeling. We used a standard_scaler fitting it on the train and transforming the test in a way to not cause data leakage.

After this we can finally merge the categorical dataset and the numerical dataset to have the final X_train_scaled and X_val_scaled ready for modeling.

  

### 1.6 - Feature Engineering:

Only for the linear regression, seeing that the y had a 0 spike and hadn’t an acceptable normal distribution, we applied a Box-Cox transformation on the training target variable y_train and on the validation target variable y_val using the estimated lambda value from the training set. We then plotted the histogram in order to better visualize the data distribution before and after the transformation.

The result was not the best but if we remove the 0 spike it was acceptable. It is important to note that this transformation was done only for the linear regression so we created two more global variables to not directly modify the y_train, y_val variables.

  

### 1.7 - First look at predictors importance:

By applying the SelectKBest class we performed a feature selection selecting the top 5 features based on the F-regression test. The selected feature indices and names are stored in selected_features_indices and selected_features_names, respectively and they are 'airline_GO_FIRST', 'airline_Indigo', 'airline_Vistara', 'class_Economy', 'duration_y'.

This is just a method to see which are the most important predictors without training a model first so when we find our best model we will also compute a SHAP analysis to see how the predictors contribute to the final prediction and which are the most important.

  

### 2 - Models:

Before moving to modeling we should say that the metrics we chose to assess the performances of our future models are the R2 and the Root Mean Squared Error (RMSE).

  

R-squared measures the proportion of variation in the target variable that the model explains, while RMSE measures the magnitude of the error between the model's predictions and the actual values. Evaluating both metrics allows for a better understanding of the strengths and weaknesses of each model and enables the selection of the best performing one.

On the one hand, RMSE is advantageous because it maintains the same scale as the predicted variable, allowing for a more intuitive interpretation of the error in the model's predictions. On the other hand, R-squared provides a means of comparing the performance of different models, as it ranges from 0 to 1, with higher values indicating a better fit to the data.

  

### 2.1 - Linear Regression:

Now we start diving into the actual models and the first one we performed is the Linear Regression and we fitted it with the scaled training features and transformed training target variable. This model uses the method of minimizing the sum of squared error to determine the best line of fit.

The linear regression model yielded impressive results with an R2 of 0.878375 and a RMSE of 0.037543. Note that this RMSE seems very low, but this is because of the transformation, in real values the error of this model would be far more big.

 
### 2.2 - Random Forest:

Our second model was an ensemble one: Random Forest Regression. For this model as for the others to come we expect a tradeoff between performance and interpretability, in fact we will lose a little bit of the second to try and have a better model for better predictions.

The Random Forest model consists of multiple trees whose training set is evaluated through the Bagging technique. The Bagging algorithm iteratively evaluates a different sample with replacement of the same original training set for each tree.

As parameters we kept it simple, since we will do a gridsearch only on our best model, increasing only the number of estimators to 300 since we noticed that with more trees, obviously, the performance increased.

As expected, this model achieved much better results than the previous ones, with an R2 of 0.98522 and a RMSE of 2761.451831

  

### 2.3 - XGBoost with D-matrices

XGBoost stands for eXtreme Gradient Boosting. It's a machine learning algorithm based on the gradient boosting framework, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees. XGBoost is known for its speed and performance.

A DMatrix is an internal data structure in XGBoost that is used to store the input data in a format that is optimized for both memory efficiency and training speed. It stands for "Dense Matrix" and it's a way to store the data in a form that makes it as fast and efficient as possible for XGBoost to work with. In this way our training and deploying of the model will be much faster than a normal xgboost.

  

### 2.3.1 - Hyperparameter tuning for XGBoost:

To find the best parameters we couldn’t use any traditional methods such as gridsearchCV or randomizedsearchCV since they are part of the sci-kit learn package. So to find the best model we made use of optuna;

Optuna is a hyperparameter optimization framework, and its goal is to find the optimal set of hyperparameters that will maximize or minimize a specific objective function. It is not a grid search method, which systematically tries out a specific set of hyperparameters, rather it's a library for Bayesian optimization, which is a more efficient method for hyperparameter tuning.

It works with an objective function that is to maximize or minimize a score that is returned by the function itself. In our case we maximized the r2 score on the validation set. Then there is a Hyperparameter space where we put our intervals of hyperparameters. And finally the optimization: Optuna uses a process of exploration and exploitation to try out different sets of hyperparameters, guided by the results of previous trials. The aim is to find the set of hyperparameters that gives the best score for the objective function.

  

After finding the best hyperparameters we fit the model through xgb.train and we found an optimal number of “rounds” to arrive at a certain performance that is 800. Our final model works very well achieving an R2 of 0.9880 and an RMSE of 2471.54

  

### 2.4 - Histgradient Boosting

HistGradientBoostingRegressor is a gradient boosting algorithm introduced in scikit-learn, specifically designed for handling large datasets efficiently. It combines the principles of gradient boosting with histogram-based gradient boosting to achieve fast and accurate predictions. Instead of using individual data points, HistGradientBoostingRegressor works with histograms of features. It discretizes the input features into bins and constructs histograms that represent the distribution of the feature values. This technique improves training and prediction speed, especially for high-dimensional datasets such as ours. We gridsearched this method fastly but it didn’t give good results such as xgboost so we didn’t include the gridsearch in our code.

The histgradient boost regressor without gridsearch arrived at an R2 of roughly 0.97 with an RMSE of 3945

  

### 2.5 - Catboost

As we have already said at the beginning of the report, by having many categorical variables we faced the challenge of encoding them to utilize conventional algorithms. However, we came across Catboost, an algorithm that offered an alternative approach. Unlike many other algorithms, Catboost eliminates the need for explicit encoding of categorical variables before utilization.

  

Catboost adopts a unique methodology, aiming to comprehend the influence of each categorical variable on the target variable. Instead of relying on traditional one-hot encoding, it calculates a numerical value that serves as the encoded representation of the categorical variable. This approach enables Catboost to leverage the original categorical data directly, potentially enhancing accuracy.

  

As for the actual result, the algorithm achieved an R2 of 0.977294 and a RMSE of 3424.3. Of course we trained the model using the original categorical variables and not the encoded ones.

  

### 2.6 - Neural network

A neural network is a computational model inspired by the structure and function of biological neural networks, such as the human brain. It is composed of interconnected nodes called artificial neurons or "units" that are organized into layers. Each unit takes inputs, applies weights to those inputs, and passes the result through an activation function to produce an output.

Here we can use Optuna to search for an acceptable neural network infrastructure and after 50 tries we used an infrastructure with 5 layers.

It is important to note that to use pytorch also X_train, X_val, y_train and y_torch must be transformed into torch items.

Unfortunately the neural network wasn’t able to perform well with an r2 of -0.97 and rmse of more than 30000. Due to high computational time cost we couldn’t find a better model. Also, we would like to add that this was just an experiment since with all these dummies and categorical variables we don’t expect the neural network (also finely built) to outperform our best model.

  

### 3 - Results and conclusions

As we saw previously we can see that our best models both in terms of performance and in terms of speed is the Xgboost and an aspect that helped this algorithm to be the best are the transformation of train and test set into d-matrices.

The second best model would be the hist gradient boosting. Also if the random forest achieved a higher R2 we consider the speed of algorithms to be important as well if not more important. This is a huge dataset so we need a fast algorithm to achieve a good score in an acceptable time.

  

### 4 - Main takeaways

Our main takeaways from this project are the ability to work with heavy datasets, the selection of the variables and how to compare models not only by numbers but by efficiency and performance.
