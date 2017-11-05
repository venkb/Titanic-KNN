#K-NN implementation
library(mice) #to impute missing data
library(caTools) #to split data into test and training set
library(class) #to build knn classifier

#setwd(Pick the folder that contains train.csv)
titanic_full = read.csv('train.csv')

#pick the features of interest
titanic = titanic_full[,c(2,3,5,6,7,8,10)]
str(titanic)

#model the features appropriately
titanic$Survived = factor(titanic$Survived)
titanic$Pclass = factor(titanic$Pclass)
titanic$SibSp = factor(titanic$SibSp)
titanic$Parch = factor(titanic$Parch)
titanic$Sex = factor(titanic$Sex, levels = c('male', 'female'), labels = c(0,1))

#impute missing values
md.pattern(titanic)
impute_titanic = mice(data = titanic,
                      m = 1,
                      maxit = 3)
imputed_titanic = complete(impute_titanic,1)

#scale continuous features
imputed_titanic$Age = scale(imputed_titanic$Age)
imputed_titanic$Fare = scale(imputed_titanic$Fare)

#create training and test sets
set.seed(5)
split = sample.split(Y = imputed_titanic$Survived,
                     SplitRatio = 0.8)
training_set = subset(imputed_titanic, split == TRUE)
test_set = subset(imputed_titanic, split == FALSE)

#build the knn classifier and predict the test set at the same time
y_pred = knn(train = training_set[,-1],
             test = test_set[,-1],
             cl = training_set[,1],
             k = 5)

#build the confusion matrix
cm = table(test_set[,1], y_pred)
cm
