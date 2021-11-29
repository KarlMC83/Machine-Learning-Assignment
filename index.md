---
title: "Machine Learning Assignment"
author: "Karl Melgarejo Castillo"
date: "28/11/2021"
output: 
  html_document:
    keep_md: true
---

## Executive summary

In this document I analyze data from sensors that measures how a weight-lifting exercise is performed in order to find the best predicting model of the manner in which the exercise is performed by using machine learning technics. I conclude that the best model uses a Random Forest algorithm with ten Principal Components as regressors. 

## 1. Analysis

In this section I describe the analysis made to find the best predicting model.

### 1.1 Exploratory analysis
First I explore the structure of the *training* and the *testing* data base with the function *str*. As it is shown below, the *training* data has 19622 observations and 160 variables; while the *testing* data has 20 observations and 160 variables.



| Dimensions / Data set     |       Training       |        Testing       | 
| :---                      |        :----:        |        :----:        | 
| Number of observations    | 19622 |  20 | 
| Number of variables       | 160 |  160 | 


The variable *classe* was used to record the manner in which the exercise is performed and appears only in the *training* data set. As you can see below, it has five classes: class A: exactly according to the specification; class B: throwing the elbows to the front; class C: lifting the dumbbell only halfway; class D: lowering the dumbbell only halfway; and class E: throwing the hips to the front.


```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```
On the other hand, information was recorded from acceleration, gyroscope and magnetometer on the belt, forearm, arm, and dumbbell for the six participants in this experiment; and their coordinates (x, y, and z axis) were registered in both data sets. We can see below the 40 variables used to record this information, which are complete (i.e. there is no NAs):


```
##  [1] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
##  [4] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
##  [7] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [10] "magnet_belt_z"        "total_accel_arm"      "gyros_arm_x"         
## [13] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [16] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [19] "magnet_arm_y"         "magnet_arm_z"         "total_accel_dumbbell"
## [22] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [25] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [28] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [31] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [34] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [37] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [40] "magnet_forearm_z"
```
The rest of variables such as the mean, standard deviation, etc. for the pitch, yaw, etc. are not complete (i.e. they have many NAs) which makes them not suitable for regressors. 

### 1.2 Machine learning model and results

I estimated five (5) models to predict the variable *classe* by using the 40 regressors shown in the previous section, which are the coordinates of the acceleration, gyroscope and magnetometer on the belt, forearm, arm, and dumbbell. 

To reduce the number of regressors and avoid over-fitting problems, I used Principal Components Analysis (PCA) to estimate ten components that capture 80% of the variance of these regressors, which will be used in three of the five models instead of the 40 original regressors.

Regarding cross validation, I used a 10-fold cross validation in all of the five models. As training method, I used *Random Forest*, *Gradient Boosting*, and *Linear Discriminant Analysis*. In the first one, I used only the 10 principal components as regressors; and in the remaining two, I used both, the principal components and the 40 regressors.

It is important to mention that the *training* data set was divided in two new sets with the function *createDataPartition*, in order to train these models in one of these two sets and then test their accuracy in the other data set. In this way we will be able to estimate the out-of-sample errors.

The results are shown in the table below:
 

```
## Warning: package 'caret' was built under R version 4.1.1
```

```
## Loading required package: ggplot2
```

```
## Loading required package: lattice
```


| Model                        |       Regressors        |  Accuracy on test data    |     Out of sample error     | 
| :---                         |          :----:         |        :----:             |         :----:              | 
| Random Forest                | 10 Principal components |0.9447749 |0.0552251 | 
| Gradient Boosting Machine    | 10 Principal components |0.6832625 |0.3167375 | 
| Gradient Boosting Machine    | Non-PCA: 40 Regressors  |0.9082413|0.0917587| 
| Linear discriminant analysis | 10 Principal components |0.3906542 |0.6093458 | 
| Linear discriminant analysis | Non-PCA: 40 Regressors  |0.6457094|0.3542906| 

As it is shown in the table above, two models have the highest accuracy ratio: the Random Forest model with 10 principal components as regressors, with 94.5% of accuracy; and the Gradient Boosting Machine model with the original 40 regressors, with 90.8% of accuracy. These accuracy ratios were calculated with the *confusionMatrix* function and using the *classe* variable in the second division of the training data set (a sort of a testing data set).

With these accuracy ratios, it is possible to estimate the out of sample error that can be achieved with a new data set, which are 5.5% and 9.2% with the Random Forest and the Gradient Boosting Machine models, respectively.

Due to the high accuracy ratio and less number of regressors, I selected the Random Forest model to predict the 20 different test cases. The result of the estimated Random Forest model is shown below:


```
## Random Forest 
## 
## 13737 samples
##    10 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 12364, 12364, 12364, 12362, 12363, 12364, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa    
##    2    0.9389253  0.9226945
##    6    0.9337558  0.9161389
##   10    0.9235633  0.9032575
## 
## Accuracy was used to select the optimal model using the largest value.
## The final value used for the model was mtry = 2.
```

## 2. Conclusion

A Random Forest algorithm, with ten Principal Components of the forty variables that measures the axis of the acceleration, gyroscope and magnetometer on the belt, forearm, arm, and dumbbell, is found to be the best model to predict the manner in which the exercise is performed.

This model has an estimated out-of-sample accuracy and out-of-sample error of 94.5% and 5.5%, respectively.   



