---
title: "Practical machine learning assignment"
author: "Sho Kubori"
date: "August 24, 2015"
output: 
  html_document: 
    number_sections: yes
---

# Introduction

 Using devices such as Nike FuelBand it is now possible to collect a large amount of data about personal activity relatively inexpensively. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 

 In this project, the goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

The training data were taken from below.
- https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data were taken from below.
- https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

# Loading and cleaning data

The data were loaded from a csv file.


```r
setwd("D:/workspace/R/Practical_Machine_Learning")
df <- read.csv("pml-training.csv", stringsAsFactors = F, header = T, colClasses = NA)
```

The data were cleaned since some variables mostly consisted of N/A, which are not considered to contribute to this analysis.


```r
df$X <- NULL
df <- transform(df, user_name = as.factor(user_name))
df <- transform(df, classe = as.factor(classe))

# sensor signals
sscols <- 7:158
for(i in sscols){
    df[,i] <- as.numeric(df[,i])
}

# data with many N/A removed
nas <- sapply(df, function(x) sum(is.na(x)))
table(nas)/ncol(df) # 59 vars good
```

```
## nas
##           0       19216       19217       19218       19220       19221 
## 0.371069182 0.421383648 0.006289308 0.006289308 0.006289308 0.025157233 
##       19225       19226       19227       19248       19293       19294 
## 0.006289308 0.025157233 0.012578616 0.012578616 0.006289308 0.006289308 
##       19296       19299       19300       19301       19622 
## 0.012578616 0.006289308 0.025157233 0.012578616 0.037735849
```

```r
df <- df[,which(nas < 19216)]
sscols <- 7:58
```

# Model selection

Random forest model for this analysis for the following 3 reasons.

1. It can deal with multiple class outcome varables.
   If response varables are categorical ones, using `method="glm"` in `train` function is an option. However, `glm` can only deal with outcome factor variables with two categories.
2. It has high precision
   Because multiple decision trees are constructed and aggregated by voting, the prediction is precise.
3. It does feature selection
   Bagging, a similar method, might create decision trees some of which have high correlation and increase variance of the model unnecessarily. Random forest method does feature selection, generates decision trees with differet features and supresses variance.
   
 Since random forest package has build-in cross validation feature, cross validation wasn't performed. Instead, parameters mtry(Number of variables randomly sampled as candidates at each split) and ntree(Number of trees to grow) were tuned.


```r
library(caret)
library(randomForest)
library(doParallel)
registerDoParallel(cores=1)
set.seed(1234)

dfnew <-df[,c(sscols, 59)] # 59:classe
# a parameter was tuned
trf.mtry <- tuneRF(dfnew[,-53], dfnew[,53],stepFactor = 1)[1]
```

![plot of chunk model1](figure/model1-1.png) 

```r
modelFit <- randomForest(classe ~ ., method="rf", data=dfnew,
                  mtry=trf.mtry, ntree=100)
```


```r
print(modelFit)
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = dfnew, method = "rf",      mtry = trf.mtry, ntree = 100) 
##                Type of random forest: classification
##                      Number of trees: 100
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.31%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 5576    3    0    0    1 0.0007168459
## B   11 3781    5    0    0 0.0042138530
## C    0   11 3408    3    0 0.0040911748
## D    0    0   19 3195    2 0.0065298507
## E    0    0    2    3 3602 0.0013861935
```

```r
plot(modelFit)
```

![plot of chunk model2](figure/model2-1.png) 

The estimated error rate was 0.46%. The graph shows how the error rate changes as ntree increases. Prior to the analysis the number of ntree was optimized. The error rate saturated at 100 as seen on the graph.

# Model application to test data

The model was applied to the test data. The test data was cleaned by the same method as the training data.


```r
test <- read.csv("pml-testing.csv", stringsAsFactors = F, header = T, colClasses = NA)

test$X <- NULL
test <- transform(test, user_name = as.factor(user_name))

# sensor signals
sscols <- 7:158
for(i in sscols){
    test[,i] <- as.numeric(test[,i])
}
test <- test[,which(nas < 19216)]
sscols <- 7:58
testnew <- test[,c(sscols, 59)]
```

Here is the result predicted by the model build above.


```r
predict(modelFit, testnew[,-53])
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

# Acknowledgement

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. They have been very generous in allowing their data to be used for this project. 
