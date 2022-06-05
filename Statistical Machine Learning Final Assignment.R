#### Question 5 

data=read.csv("data_rotten_tomatoes_review.csv")
str(data) #check the class of each variable 
library(caret)
data$class=factor(data$class) 
range(data[,-c(1,81)]) #variables are measured in different magnitudes so need to standardise the data
data[,-c(1,81)]=scale(data[,-c(1,81)])
data=data[,-1]
library(doParallel)
cl <- makeCluster(6,setup_strategy = "sequential") 
registerDoParallel(cl)

train_val=createDataPartition(data$class,p=0.7,list=FALSE)
data_train=data[train_val,]
data_test=data[-train_val,]

# 3 folds cross validation is applied when tuning models when the procedure is repeated 10 times for each tuned model
kfoldscvtrain=trainControl(method="repeatedcv",number=3,repeats=5)

#Used 3 cost parameters since computational time was slow on my laptop
set.seed(4573)
tune_grid_for_svm=expand.grid(C=c(5,50,100),
                              sigma=c(0.033,0.066,0.1))

svm_grbf=train(class~.,data=data_train,method="svmRadial",trControl=kfoldscvtrain,
               tuneGrid=tune_grid_for_svm)

svm_grbf

set.seed(6473)
tune_grid_for_random_forest=expand.grid(mtry=c(5,20,40,79))

random_forest=train(class~.,data=data_train,method="rf",trControl=kfoldscvtrain,
                    tuneGrid=tune_grid_for_random_forest)
random_forest

set.seed(6476)

tune_grid_for_logistic_reg=expand.grid(alpha=0,lambda=c(0.5,5,35,50))

logistic_reg=train(class~.,data=data_train,method="glmnet",trControl=kfoldscvtrain,
                    tuneGrid=tune_grid_for_logistic_reg)

logistic_reg

stopCluster(cl)

#Bagging/Random Forest (mtry=20) seems to have the highest validation accuracy of 70.78%

set.seed(4352)
comp=resamples(list(svm_radial= svm_grbf,glmnet=logistic_reg,rf=random_forest))
summary(comp)

set.seed(5352)
class_hat=predict(random_forest,newdata=data_test)
confusionMatrix(class_hat,data_test$class,positive = 'positive')
