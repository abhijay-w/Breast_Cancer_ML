# Breast_Cancer_ML
Breast cancer is one of the most severe cancers. It has taken hundreds of thousands of lives every year. Early prediction of breast cancer plays an important role in successful treatment and saving the lives of thousands of patients every year. However, the conventional approaches are limited in providing such capability. 

The recent breakthrough of data analytics and data mining techniques have opened a new door for healthcare diagnostic and prediction. Machine learning methods for diagnosis can significantly increase processing speed and on a big scale can make the diagnosis significantly cheaper.

## Dataset
Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. The dataset is taken from the [UCI Machine learning website](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)).Take cleaned dataset from [Kaggle](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data).

Attribute Information:

1. ID number
2. Diagnosis (M = malignant, B = benign)


Ten real-valued features are computed for each cell nucleus:

* radius (mean of distances from center to points on the perimeter)
* texture (standard deviation of gray-scale values)
* perimeter
* area
* smoothness (local variation in radius lengths)
* compactness (perimeter\^2 / area - 1)
* concavity (severity of concave portions of the contour)
* concave points (number of concave portions of the contour)
* symmetry
* fractal dimension (\"coastline approximation\" - 1)

The mean, standard error and \"worst\" or largest (mean of the three\ largest values) of these features were computed for each image,\ resulting in 30 features. For instance, field 3 is Mean Radius, field\ 13 is Radius SE, field 23 is Worst Radius.

All feature values are recoded with four significant digits.

```R
setwd("Path to workspace")
getwd()

#Loading Packages
library(caret)
library(ggfortify)
library(dplyr)
library(tidyverse)
library(magrittr)

#Loading dataset
df <- read.csv("data.csv")

# the 33 column is not right
df[,33] <- NULL

# This is defintely an most important step:  
# Check for appropriate class on each of the variable.  
glimpse(df)
```

![Glimpse image](/img/1.jpg)

So we have 569 observations with 32 variables. Ideally for so many variables, it would be appropriate to get a few more observations.

## Tidy the data

Tidy data is a standard way of mapping the meaning of a dataset to its structure. A dataset is messy or tidy depending on how rows, columns and tables are matched up with observations, variables and types. In tidy data:

1.Each variable forms a column.

2.Each observation forms a row.

3.Each type of observational unit forms a table.

```R  
  df$diagnosis <- as.factor(df$diagnosis)

  #Missing values
  map_int(df, function(.x) sum(is.na(.x)))

  round(prop.table(table(df$diagnosis)), 2)
```

![](/img/2.jpg)

As we see there are no missing values.

In the case that there would be many missing values, we would go on the transforming data for some appropriate imputation.

**Class distribution: 357 benign, 212 malignant**
The response variable is slightly unbalanced.

###  Understanding the data

This is the circular phase of our dealing with data. This is where each of the transforming, visualizing and modeling stage reinforce each other to create a better understanding.

Let's check for correlations.

For an analysis to be robust it is good to remove multicollinearity (i.e remove highly correlated predictors). Because multicollinearity reduces the precision of the estimate coefficients, which weakens the statistical power of the regression model. And it might become difficult to trust the p-values to identify independent variables that are statistically significant.

``` R
#Checking correlations
df_corr <- cor(df %>% select(-id, -diagnosis))
corrplot::corrplot(df_corr, order = "hclust", tl.cex = 1, addrect = 8)
```
![correlationsmatrix](/img/3.png)


There are few variables that are highly correlated such as perimeter_worst with area_mean,area_worst with area_mean,radius_mean with perimeter_mean and many more variables. 

On the next step, we will remove the highly correlated ones using the caret package. Because when we have two independent **variables** that are very highly **correlated**, we **should remove** one of them because it leads to multicollinearity.

## Transformation and preprocessing
 ```R
# The findcorrelation() function from caret package remove highly correlated predictors
# based on whose correlation is above 0.9. This function uses a heuristic algorithm 
# to determine which variable should be removed instead of selecting blindly
df2 <- df %>% select(-findCorrelation(df_corr, cutoff = 0.9))
ncol(df2)
 ```
**22**. So our new data frame df2 is 10 variables shorter.


### Using PCA Algorithm

PCA is a type of linear transformation on a given data set that has values for a certain number of variables (coordinates) for a certain amount of spaces. This linear transformation fits this dataset to a new coordinate system in such a way that the most significant variance is found on the first coordinate, and each subsequent coordinate is orthogonal to the last and has a lesser variance. In this way, you transform a set of x correlated variables over y samples to a set of p uncorrelated principal components over the same samples.

Let's first go on an unsupervised analysis with a PCA analysis. To do so, we will remove the id and diagnosis variable, then we will also
scale and center the variables.

**First applying on the original dataset with all the correlated
variables**

```R
#PCA analysis
preproc_pca_df <- prcomp(df %>% select(-id, -diagnosis), scale = TRUE, center = TRUE)
summary(preproc_pca_df)
```
![](/img/4.jpg)

A principal component is a normalized linear combination of the original
predictors in a data set. Let's say we have a set of predictors as

X¹, X²\...,X^p^

The principal component can be written as:

Z¹ = Φ¹¹X¹ + Φ²¹X² + Φ³¹X³ + \.... +Φ^p^¹X^p^

```R
# Calculate the proportion of variance explained
pca_df_var <- preproc_pca_df$sdev^2
pve_df <- pca_df_var / sum(pca_df_var)
cum_pve <- cumsum(pve_df)
pve_table <- tibble(comp = seq(1:ncol(df %>% select(-id, -diagnosis))), pve_df, cum_pve)

#Ploting
ggplot(pve_table, aes(x = comp, y = cum_pve)) + 
  geom_point() + 
  geom_abline(intercept = 0.95, color = "red", slope = 0) + 
  labs(x = "Number of components", y = "Cumulative Variance")
```

![](/img/graph1.png)


**With the original dataset, 95% of the variance is explained with 10
PC's.**

**First principal component** is a linear combination of original predictor variables which captures the maximum variance in the data set. It determines the direction of highest variability in the data. Larger the variability captured in first component, larger the information captured by component. No other component can have variability higher than first principal component.

The first principal component results in a line which is closest to the data i.e. it minimizes the sum of squared distance between a data point and the line.

**Second principal component** (Z²) is also a linear combination of original predictors which captures the remaining variance in the data set and is uncorrelated with Z¹. In other words, the correlation between first and second component should is zero.

```R
#Most influential variables for the first 2 components
pca_df <- as_tibble(preproc_pca_df$x)
ggplot(pca_df, aes(x = PC1, y = PC2, col = df$diagnosis)) + geom_point()
```
![](/img/graph2.png)


As expected, the first 2 components managed to separate the diagnosis quite well. Lots of potential here.

For a more detailed analysis of what variables are the most influential in the first 2 components, we can use the ggfortify library.

```R
autoplot(preproc_pca_df, data = df,  colour = 'diagnosis',
         loadings = FALSE, loadings.label = TRUE, loadings.colour = "blue")
```
![](/img/graph3.png)

We will do the same exercise with our second df, the one where we
removed the highly correlated predictors.

```R
#For df2
preproc_pca_df2 <- prcomp(df2, scale = TRUE, center = TRUE)
summary(preproc_pca_df2)
```
![](/img/5.jpg)

We can see that the 1st Principle component explains 46.69% of the
variance and the 2nd Principle component explains 20.38%.

```R
pca_df2_var <- preproc_pca_df2$sdev^2

# proportion of variance explained
pve_df2 <- pca_df2_var / sum(pca_df2_var)
cum_pve_df2 <- cumsum(pve_df2)
pve_table_df2 <- tibble(comp = seq(1:ncol(df2)), pve_df2, cum_pve_df2)

ggplot(pve_table_df2, aes(x = comp, y = cum_pve_df2)) + 
  geom_point() + 
  geom_abline(intercept = 0.95, color = "red", slope = 0) + 
  labs(x = "Number of components", y = "Cumulative Variance")
```

Plotting the number of components and cumulative variance graph.
![](/img/graph4.png)

**We can see that In this case, around 8 PC's explained 95% of the
variance.**

### Using LDA Algorithm

Linear Discriminant Analysis is a well-established machine learning technique and classification method for predicting categories. Its main advantages, compared to other classification algorithms such as neural networks and random forests, are that the model is interpretable, and that prediction is easy. Linear Discriminant Analysis is frequently used as a dimensionality reduction technique for pattern recognition or classification of data and machine learning.

LDA determines group means and computes, for each individual, the probability of belonging to the different groups. The individual is then affected to group with the highest probability score.

The lda() outputs contain the following elements:

* Prior probabilities of groups: the proportion of training observations
in each group.

* Group means: group center of gravity. Shows the mean of each variable in
each group.

* Coefficients of linear discriminants: Shows the linear combination of
predictor variables that are used to form the LDA decision rule.

Breast cancer dataset have many different classes and categories 

1.  We are taking the different classes of our dataset. Here package
    MASS ("Modern Applied Statistics with S") contains LDA and QDA.

```R
#LDA
preproc_lda_df <- MASS::lda(diagnosis ~., data = df, center = TRUE, scale = TRUE)
preproc_lda_df
```

 -\> Now after reducing the dimensions of our dataset we make
    predictions using the predict functionality.

 -\> the predict functions give us the predicted classes of the
      observations.

2. Here the predict function is used to predict the classes for
preproc_lda_df. 

3. cbind is used to combine two columns and as data frame is used to
convert to data frame and then using the glimpse function from dplyr package we print the output of
the prediction we made in our last step

```R
# Making a df out of the LDA for visualization purpose.
predict_lda_df <- predict(preproc_lda_df, df)$x %>% 
  as_tibble() %>% 
  cbind(diagnosis = df$diagnosis)

glimpse(predict_lda_df)
```

![](/img/6.jpg)

-\> LD1: it is a linear discriminant function. It achieves the maximal separation of our class. Lda creates one or more linear combinations of predictors , creating a new variable for each function.

## Modelling the Data

Now we are going to develop our model. We will start by first splitting our dataset into two parts using caret; one as training set for the model, and the other as a test set to validate the predictions that the model will make. If we omit this step, the model will be trained and tested on the same dataset, and it will underestimate the true error rate, a phenomenon known as overfitting.

```R
#Model the data
set.seed(1815)
df3 <- cbind(diagnosis = df$diagnosis, df2)
df_sampling_index <- createDataPartition(df3$diagnosis, times = 1, p = 0.8, list = FALSE)
df_training <- df3[df_sampling_index, ]
df_testing <-  df3[-df_sampling_index, ]
df_control <- trainControl(method="cv",
                           number = 15,
                           classProbs = TRUE,
                           summaryFunction = twoClassSummary) 
```

We would partition our final data frame into training and testing sets.We use 80% of the data for training while remaining 20% for testing. We also apply cross-validation technique to resample the data at least 15 times.

### Logistic Regression

Logistic regression is a type of classification method in statistical learning. It models the probability that Y (Y=p(X)) belongs to a particular class using the logistic function, where the coefficients are estimated using the maximum likelihood function. The response is a probability estimate that indicates the odds in which an observation will belong to a certain class. For binary classes, typically, the decision boundary is 0.5. In this case, if the probability is > 0.5, then the prediction is M, the diagnosis is malignant.

**Implementation of Logistic Regression**

First model is doing logistic regression on df2, the data frame where we took away the highly correlated variables.

```R
#Logistic Regression
model_logreg_df <- train(diagnosis ~., data = df_training, method = "glm", 
                         metric = "ROC", preProcess = c("scale", "center"), 
                         trControl = df_control)

prediction_logreg_df <- predict(model_logreg_df, df_testing)
cm_logreg_df <- confusionMatrix(prediction_logreg_df, df_testing$diagnosis, positive = "M")
cm_logreg_df
```
Output
```
Confusion Matrix and Statistics

          Reference
Prediction  B  M
         B 67  0
         M  4 42
                                          
               Accuracy : 0.9646          
                 95% CI : (0.9118, 0.9903)
    No Information Rate : 0.6283          
    P-Value [Acc > NIR] : <2e-16          
                                          
                  Kappa : 0.9257          
                                          
 Mcnemar's Test P-Value : 0.1336          
                                          
            Sensitivity : 1.0000          
            Specificity : 0.9437          
         Pos Pred Value : 0.9130          
         Neg Pred Value : 1.0000          
             Prevalence : 0.3717          
         Detection Rate : 0.3717          
   Detection Prevalence : 0.4071          
      Balanced Accuracy : 0.9718          
                                          
       'Positive' Class : M               
```

### Random Forest Algorithm

Random forest is a tree-based algorithm which involves building several
trees (decision trees), then combining their output to improve
generalization ability of the model.

While creating random trees it split into different nodes or subsets.
Then it searches for the best outcome from the random subsets. This
results in the better model of the algorithm. 

Firstly, we create train and test folds.Then we fit a model using the train function. We use the metric ROC (It is a plot of the **True Positive Rate (on the y-axis)** versus
the **False Positive Rate (on the x-axis)** for every possible classification threshold. )

```R
#Random Forest
model_rf_df <- train(diagnosis ~., data = df_training,
                     method = "rf", 
                     metric = 'ROC', 
                     trControl = df_control)

prediction_rf_df <- predict(model_rf_df, df_testing)
cm_rf_df <- confusionMatrix(prediction_rf_df, df_testing$diagnosis, positive = "M")
cm_rf_df
```
Output
```
Confusion Matrix and Statistics

          Reference
Prediction  B  M
         B 70  1
         M  1 41
                                          
               Accuracy : 0.9823          
                 95% CI : (0.9375, 0.9978)
    No Information Rate : 0.6283          
    P-Value [Acc > NIR] : <2e-16          
                                          
                  Kappa : 0.9621          
                                          
 Mcnemar's Test P-Value : 1               
                                          
            Sensitivity : 0.9762          
            Specificity : 0.9859          
         Pos Pred Value : 0.9762          
         Neg Pred Value : 0.9859          
             Prevalence : 0.3717          
         Detection Rate : 0.3628          
   Detection Prevalence : 0.3717          
      Balanced Accuracy : 0.9811          
                                          
       'Positive' Class : M
```

Plot:

```R
plot(model_rf_df)
```
![](/img/graph5.png)


X-axis consists of the hyperparameter which is the number of randomly selected variables used at each split .

Y -axis denotes the ROC Cross validation values (Cross Validation is a technique which involves reserving a particular sample of a data set on which we do not train the model. )

From the plot we can infer that values are higher of ROC if the number of randomly selected predictors is less. We see a negative slope as the number of randomly selected predictors increase.

```R
plot(model_rf_df$finalModel)
```
![](/img/graph6.png)  

This plot depicts The test error which is displayed as a function of the number of trees.

Each coloured line corresponds to a different value, the number of predictors available for splitting at each interior tree node.

The test error is lowest around 400 trees and the error values are largely stable after 250 trees.

Now we look at a plot of Mean Decrease Gini vs variables having the most predictive power.

```R
randomForest::varImpPlot(model_rf_df$finalModel, sort = TRUE, 
                         n.var = 10, main = "The 10 variables with the most predictive power")
```
![](/img/graph7.png) 

MeanDecreaseGini : GINI is a measure of node impurity. Think of it like this, if you use this feature to split the data, how pure will the nodes be. Highest purity means that each node contains only elements of a single class. Assessing the decrease in GINI when that feature is omitted leads to an understanding of how important that feature is to split the data correctly.

From this plot we can infer that higher the value of mean decrease Gini higher the predictive power.

### K-Nearest Neighbor (K-NN)

A k-nearest-neighbor algorithm, often abbreviated k-nn, is an approach to data classification that estimates how likely a data point is to be a member of one group or the other depending on what group the data points nearest to it are in.

The k-nearest-neighbor is an example of a \"lazy learner\" algorithm, meaning that it does not build a model using the training set until a query of the data set is performed.

You can't pick any random value for k. The whole algorithm is based on the k value. Even small changes to k may result in big changes. Like most machine learning algorithms, the K in KNN is a hyperparameter. You can think of K as a controlling variable for the prediction.

**Implementation**

We have created a train and test folds, then we fitted a model using the train function. Here we have used the metric ROC i.e., a plot of the True Positive Rate on the Y-axis versus the False Positive Rate on the X-axis for every possible classification threshold.

```R
#KNN
model_knn_df <- train(diagnosis ~., data = df_training, 
                      method = "knn", 
                      metric = "ROC", 
                      preProcess = c("scale", "center"), 
                      trControl = df_control, 
                      tuneLength =31)

plot(model_knn_df)
```

![](/img/graph8.png) 

-   X-axis consists of the neighbours which means the different values
    of k.

-   Y -axis denotes the ROC Cross validation values and we have used
    this here because it is a technique which involves reserving a
    particular sample of a data set on which we do not train the model.

-   Here, we have **used ROC** to select the optimal model using the
    largest value. That is why **31 times** resampling the results
    across the **tuning parameters** has been done.

So, the **AUC of ROC** is attaining the peak value when the neighbours is 15 that means we get an optimal **K-NN model at k=15.**

```R
prediction_knn_df <- predict(model_knn_df, df_testing)
cm_knn_df <- confusionMatrix(prediction_knn_df, df_testing$diagnosis, positive = "M")
cm_knn_df
```

When we get the data, after data cleaning, pre-processing and wrangling, the first step we do is to feed it to an outstanding model and of course, get output in probabilities.

For better the effectiveness, better the performance confusion matrix comes into the limelight. Confusion Matrix is a performance measurement for machine learning classification.

It is extremely useful for measuring Recall, Precision, Specificity, Accuracy and most importantly AUC-ROC Curve.

```
Confusion Matrix and Statistics

          Reference
Prediction  B  M
         B 71  3
         M  0 39
                                          
               Accuracy : 0.9735          
                 95% CI : (0.9244, 0.9945)
    No Information Rate : 0.6283          
    P-Value [Acc > NIR] : <2e-16          
                                          
                  Kappa : 0.9423          
                                          
 Mcnemar's Test P-Value : 0.2482          
                                          
            Sensitivity : 0.9286          
            Specificity : 1.0000          
         Pos Pred Value : 1.0000          
         Neg Pred Value : 0.9595          
             Prevalence : 0.3717          
         Detection Rate : 0.3451          
   Detection Prevalence : 0.3451          
      Balanced Accuracy : 0.9643          
                                          
       'Positive' Class : M                                                 
```

### SVM (Support Vector Machines) with PCA

SVM or Support Vector Machine is a supervised learning algorithm which has the ability to produce a decision boundary (hyperplane) that can segregate an n-dimensional space into classes so that we can easily put the new data point in the correct category. SVM chooses the extreme points/vectors that help in creating the hyperplane which are known as support vectors. The hyperplane thus produced can be used for classification, regression, and other tasks like outliers detection.

Before training and testing the dataset, we apply PCA to our model. The trainControl function specifies number of parameters in the model. The object outputted from trainControl is given as argument in train.

```R
#SVM with PCA 
set.seed(1815)
df_control_pca <- trainControl(method="cv",
                               number = 15,
                               preProcOptions = list(thresh = 0.9), # threshold for pca preprocess
                               classProbs = TRUE,
                               summaryFunction = twoClassSummary)

model_svm_pca_df <- train(diagnosis~.,
                          df_training, method = "svmLinear", metric = "ROC", 
                          preProcess = c('center', 'scale', "pca"), 
                          trControl = df_control_pca)

prediction_svm_pca_df <- predict(model_svm_pca_df, df_testing)
cm_svm_pca_df <- confusionMatrix(prediction_svm_pca_df, df_testing$diagnosis, positive = "M")
cm_svm_pca_df
```

Output:
```
Confusion Matrix and Statistics

          Reference
Prediction  B  M
         B 70  1
         M  1 41
                                          
               Accuracy : 0.9823          
                 95% CI : (0.9375, 0.9978)
    No Information Rate : 0.6283          
    P-Value [Acc > NIR] : <2e-16          
                                          
                  Kappa : 0.9621          
                                          
 Mcnemar's Test P-Value : 1               
                                          
            Sensitivity : 0.9762          
            Specificity : 0.9859          
         Pos Pred Value : 0.9762          
         Neg Pred Value : 0.9859          
             Prevalence : 0.3717          
         Detection Rate : 0.3628          
   Detection Prevalence : 0.3717          
      Balanced Accuracy : 0.9811          
                                          
       'Positive' Class : M               
                                          
```

### Neural Nets with LDA

A neural network is a machine learning algorithm based on the model of a human neuron. In general, Neural Network has three neurons, namely input neuron, hidden neuron, and output neuron.

Next, there are two main processes in the Neural Network that are commonly used, namely feedforward and backpropagation. Feedforward is an algorithm for calculating output values ​​based on input values, while backpropagation is an algorithm for training neural networks (changing weights) based on errors obtained from output values.

The process for calculating the output is as follows.

i.  Multiply the weights by the input of the neurons then add them up.

ii. Enter the results of the multiplication of weights by the input into
    the activation function which can be a sigmoid function.


After we get the value for the output layer, the next process is to get the error value from the output layer. This error value will later
become a parameter to change the weight value. To calculate errors there are also many methods such as MSE (Mean Squared Error), SSE (Sum of Squared Error), etc.

These 2 processes will continue to repeat until the minimum error value is reached. After the training process is deemed sufficient, the model can be used
for the classification process.

**Training & Testing the Model**

Prior to training our model we require a training and testing set. Now, as defined earlier we have used LDA for dimensionality reduction, so we will be making use of the data frame (*predict_lda_df*) obtained from there to create a training and testing set and apply Neural Network with LDA.

```R
lda_training <- predict_lda_df[df_sampling_index, ]
lda_testing <- predict_lda_df[-df_sampling_index, ]
```
While training our *nnet* model we have centered and scaled our feature values for unit independency while using the ROC curve as the metric for obtaining an optimal model. We tune our model over 10 values and use the best combination to train the final model with.

Finally, after predicting on the test set, we plot the confusion matrix
and statistics for performance analysis.

```
Confusion Matrix and Statistics

          Reference
Prediction  B  M
         B 70  1
         M  1 41
                                          
               Accuracy : 0.9823          
                 95% CI : (0.9375, 0.9978)
    No Information Rate : 0.6283          
    P-Value [Acc > NIR] : <2e-16          
                                          
                  Kappa : 0.9621          
                                          
 Mcnemar's Test P-Value : 1               
                                          
            Sensitivity : 0.9762          
            Specificity : 0.9859          
         Pos Pred Value : 0.9762          
         Neg Pred Value : 0.9859          
             Prevalence : 0.3717          
         Detection Rate : 0.3628          
   Detection Prevalence : 0.3717          
      Balanced Accuracy : 0.9811          
                                          
       'Positive' Class : M               
                                          
```

By Looking at the results we can see that neural network can perform exceptionally well even in case of loss in data during the dimensionality reduction or pre-processing steps. Also, even if a neuron is not responding or a piece of information is missing, the network can detect the fault and still produce the output. The use of LDA further strengthens our nnet model.


## Model Evaluation

```R
#Model Evaluation
model_list <- list(logisic = model_logreg_df, rf = model_rf_df, knn=model_knn_df,
                   SVM_with_PCA = model_svm_pca_df,Neural_with_LDA = model_nnetlda_df)
results <- resamples(model_list)

summary(results)

bwplot(results, metric = "ROC")
```

![](/img/graph9.png)

![](/img/7.jpg) 


The logistic has to much variability for it to be reliable. The Random Forest and Neural Network with LDA pre-processing are giving the best results. The ROC metric measure **the auc of the roc curve** of each model. This metric is independent of any threshold. Let's remember how these models result with the testing dataset. Prediction classes are obtained by default with a threshold of 0.5 which could not be the best with an unbalanced dataset like this.


|Model  | Accuracy |        
| ------------- |:-------------:|
|Neural Network with LDA |	98.23% |
|Random Forest	         |  98.23% |
|SVM with PCA dataset	   |  98.23% |
|KNN                     |  97.35% |
|Logistic regression	   |  96.46% |

## Conclusion

This project was carried out to predict the accuracy of determining
cancer at early state, after comparing five different models.We see that three models achieve accuracy of 98.23%.
