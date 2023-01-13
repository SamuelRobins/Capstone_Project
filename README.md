# Echo Park Auto Motive Model

## Business Problem
This project is made for Echo Park Automotive. They are a used car dealership. I have made a Decision Tree Regressor model. This model will process the data of used cars, their features, and their price. Aftrer learning the relationships between the details or features of the car and its selling price, it can begin to analize a car and predict its price. This has been created so that Echo Park Automotive can decide which cars to buy. If the model predicts a higher selling price than what is proposed on a prospective car, they should buy it and can flip the car. If the model predicts a lower selling price then what is proposed, then they should not buy the car.

## Data Understanding

Our Data was sourced from Kaggle (https://www.kaggle.com/datasets/rakkesharv/used-cars-detailed-dataset). It created by webscraping several used car sales websites. There are 973 entries with about 20 different features of each car. The target is the sales price and we are using the other features to predict the sales price. These features include :

Car_Name: The full name of the car which is displayed in the ad
Make: Maker of the Car
Model : Model of the Car
Make Year: Year of Manufacturing
Color : Color of the Car
Body Type : Body type of the car
Mileage Run: Total KMs the car run
No of Owners: Number of Previous Owners
Seating Capacity: Total Seating Capacity Available
Fuel Type: Fuel Type used by the car
Fuel Tank Capacity(L) : Total Fuel Capacity of the car
Engine Type : Engine Name, Model and Type
CC Displacement: Total Cubic Displacement
Transmission : Kind of Transmission
Transmission Type: Type of Transmission
Power(BHP) : Total Max Power
Torque(Nm) : Total Max Torque
Mileage(kmpl) : Average Mileage of the Car
Emission: Emission Norms of the Car
Price: Selling Price.

## Packages
I used the following packages for my modeling
import numpy as np
import scipy.stats as stats
import sklearn
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyRegressor as dr
from sklearn import metrics

## Independant Learning
Independant Learning from getjerry.com (its a car-financing website) “Color generally doesn't have an impact on the price of buying a new vehicle. However, in-demand colors like white, grey, and silver may be widely available and easier to purchase than those with unique colors. More popular colors will also hold better resale value over time since there is more demand for them."

## Data Analysis

Now lets take the data, which is a csv file, and turn it into a data frame by using an import called pandas.


<class 'pandas.core.frame.DataFrame'>
RangeIndex: 973 entries, 0 to 972
Data columns (total 20 columns):
    Column                 Non-Null Count  Dtype  
---  ------                 --------------  -----  
 0   Car_Name               973 non-null    object 
 1   Make                   973 non-null    object 
 2   Model                  973 non-null    object 
 3   Make_Year              973 non-null    int64  
 4   Color                  973 non-null    object 
 5   Body_Type              973 non-null    object 
 6   Mileage_Run            973 non-null    int64  
 7   No_of_Owners           973 non-null    object 
 8   Seating_Capacity       973 non-null    int64  
 9   Fuel_Type              973 non-null    object 
 10  Fuel_Tank_Capacity(L)  973 non-null    int64  
 11  Engine_Type            973 non-null    object 
 12  CC_Displacement        973 non-null    int64  
 13  Transmission           973 non-null    object 
 14  Transmission_Type      973 non-null    object 
 15  Power(BHP)             973 non-null    float64
 16  Torque(Nm)             973 non-null    float64
 17  Mileage(kmpl)          973 non-null    float64
 18  Emission               973 non-null    object 
 19  Price                  973 non-null    object 
dtypes: float64(3), int64(5), object(12)
memory usage: 152.2+ KB
Some of these string objects can be converted into intergers by removing commas. This will make for better processing. Lets take a look at 'Emissions' for example. The roman numerals can be converted to values. Lets first make sure that they are all roman numerals by using a method called value counts.

Some of these string objects can be converted into intergers by removing commas. This will make for better processing. Lets take a look at 'Emissions' for example. The roman numerals can be converted to values. Lets first make sure that they are all roman numerals by using a method called value counts.


### Data Prep: The First Step Is Making Numbers Where We Can

Im going to change the string to interger or float conversion to all of the columns I feel are good candidates to change. I am using the replace method to remove the elements that would cause these values to be a string. Next, I will call on a method 'astype' where I can tell the computer what kind of data type I want the selected values to be.
### Dropping Overbearing Columns
Okay, so I have eight object columns left. That is too many for my analysis. This is because I will make the identification of the objects a true false statement (Binary) so that the model can compute their relationship to the data. This will create many more columns. 

It is apparant that the column 'Car Name' has descriptions that are in other columns. That leads me to decide that it is unecessary for the model. So I am going to remove that column by calling on the .drop method. I am also going to remove the 'Color' column. This is because of the  independant learning I have completed. I am assuming that the color of the car does not have the kidn of relationship with thetarget value that we need to calculate. I am also going to drop the model, the make, the engine type. This is because they have too many values and would add 200 columns to my data frame and that would disrupt the model. 

### Individual Analysis of each features relationship with the target.

We have 15 including our target value. Let's look at the relationship between the features and our target value. I am going to do this by making an individual graph of each feature and the target value.
(Visual in notebook)
As you can see, many of the features have a linear relationship with the model. But a few don't. While it would be more simple to drop the features that aren't linear to make a linear regression model, I want my model to be very accurate and incorporate those features. So I am going to mess around and see which kind of modeling scores best.

# Modeling

Now we will split the data using an import from sklearn call 'train, test, split'. This will split the data in way so that we can use the majority of it to select, and train our model. Then, when we have developed our final model, we will use the testing data to see how accurately it can predict the target value which in this case is the price. The first step is to isolate the target variable from the data frame.

### Dummy Model

Now let's apply a dummy regressor to see our baseline. In other words, a sort of 'floor' to compare to see how much were solving for with our modeling. I imported the dummy regressor as DR for an abbreviate.

We can see that our dummy model esentially doesn't solve anything because we havent built the right model for the data. It got an R squared score of 0, which means that none of the variance in the target data is a result of the features in this models computation. This is because the dummy model simply predicts the mean as every value.

### Piping

Now I am going to do something called Piping. this is where in wrapp up certain proceadures like StandardScaler and OneHotEncoder. StandardScaler will scale all of my numeric values so that they can be properly compared to eachother to asses their impact on the target value. OneHotEncoder will turn all of my categorical values into binary columns. This will allow them to be scaled with the other values for the computation. It is like scaling for objects.

Here when I set up my pipe to work on columns, I assigned the 'pipe' that I want and after that I assigned the columns I want the pipe to transform with an array as the last argument. The array tells the transformer which columnsto transform.

Now I am going to create one last pipe, which will incorporate my Transformer, and my model into one pipe. Let's start off a linear regression to see how that scores.

The R_Squared score is an 82.N ot bad, lets try this with a Decision Tree Regressor and see how that one performs.

Now we got an R_squared score of 100. Clearly much better. Yet, we might be over fit. This is something we will address in our next steps.


### Grid

While our model may be over fit, it certainly is a much better score. Let's run it through a grid to find our best parameters for the decision tree regressor.

A grid runs the model with all of the parameter options that I create. It will then tell me which combination of parameter options score the best.

I am going to use the criterion you see below(In notebook), because there are only four I know of that are commonly used. So why not test them all. The splitter option which decides at which nodes the decision tree should split, is most commonly either best or random so I thought I would make both an option and see which one is better. From my intuition I believe that 5 will be the best amount of minimum samples to split. Thats essentialy how deap the tree goes. But I could be either to high or too low with that estimation. So (no punn intended) I am going to go two up and two down as my other opens and see which one performs the best. 

We got a R_squared score of 99. This is really a remarkable score. The tuning probably cleared out someone of the types of interpretations that made it over fit. Let's look at the best parameters for the Decision Tree Regressor.

Intresting. This means that with the specific set of data that we are using, the model runs best on it with these parameters (In grid in notebook).

Now let's take a look at the predictions our model makes based on the data from the test set. 'X_test'.

### Evaluation

So let's assign this to a variable called y_pred because its the models prediction of y which is supposed to be the target value.

Now lets get the mean squared error but let not square it so that its really just the mean error.

Now that we have the mean squared error (94217), lets return the R^2 score!

Our R_squared tells us that 92% of the varience in price is explained by the features in this model. This is a testament to the accuracy of the model.

We made a table which shows that there is an aerage of 8.4% difference amongst the real and predicted values

(Tables in Notebook)

SO you can see here that the highly accurate model predicted that the first four cars in this table would sell for significantly more than their current prices. We know its accurate because of its R_squared score. This is an example of how our Echo Park Automotive would monitize on our model. Essentially they would buy the cars for there current price in 'real values' and then flip them for the values that the model predicted. The last car on the table is predicted to sell for less than the real value. This is an example of a car the model is telling you that you should not buy.

Now we have a scatter plot (In notebook) where the x axis is the predicted price and the y axis is the actual price data of the testing set. I also added a best line of fit to show the accuracy.

## Recommendation 
We recommend that Echo Park Automotive used our model to predict the price of used cars. We recommend that they buy used cars which are selling for less than their predicted price. After buying the cars, they can then flip them for the predicted price. They will make money because they have sold cars for more than they have bought them. We also recommend that Echo Park Automotive does not buy cars which are selling for more than their predicted price.


# Conclusion

We essentially used a series of web scraped data to make a model that is capable of monetizing accuracy on a market analysis. Not every car sells for the exact price it should based on its featues. This is due to many variables which are yet to be identified. What we have done is created a way for Echo Park Automotive to accuratley predict the selling price of a used car and to capitalize on the wide range of the selling market. We have created a model that has an R_squared score of 92. That is highly accurate and gives Echo Park Automotive the confidence they need to invest based off of our model. We can see that there were many steps taken into buildign the model including data analysis, data prep, figuring out which kind of model best suits our data, piping some of our data prep into our model tuning and engaging in a grid search to create our most accurate model. What has resulted of this is guide for the business model of Echo Park Automotive. 

### Next steps
First let's acknowledge that the model is a little bit over fit to the data. This is something we would work on down the road. The main goal moving forward is to apply this concept to many other kinds of assets. There is so much more data out there that we can tap into. The last thing is that we would set a stream of data that is constantly webscraping to update our model. Market's change and shift. In todays world, you need to be constantly ready to evolve or you will get overpowered by a competitor.
