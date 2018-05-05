setwd("C:\\Users\\suman\\Desktop\\termdeposit")
bank<-read.csv("bank-additional.csv", header=TRUE,sep=';')

#find relation between independent variables and dependent variable 
boxplot(bank$y,bank$duration)
#as duration is highly corelated its remove
bank <- subset(bank, select = -c(duration))
set.seed(1235)
TrainingDataIndex <- createDataPartition(bank$y, p=0.75, list = FALSE)
train <- bank[TrainingDataIndex,]
test <-bank[-TrainingDataIndex,]
prop.table(table(train$y))



#decision tree
#use 10 fold cross validation
TrainingParameters <- trainControl(method = "cv", number = 10, repeats = 5)
#create decision tre with c5.0
DecTreeModel <- train(y ~ ., data = train, 
                      method = "C5.0",
                      trControl= TrainingParameters,
                      na.action = na.omit)

DecTreeModel
summary(DecTreeModel)
#this shows 9.9% error
#test decision tree
DTPredictions <-predict(DecTreeModel, test, na.action = na.pass)
confusionMatrix(DTPredictions, test$y)




#naive bayes
set.seed(100)
TrainingDataIndex <- createDataPartition(bank$y, p=0.75, list = FALSE)
train <- bank[TrainingDataIndex,]
test <-bank[-TrainingDataIndex,]
NBModel <- train(train[,-20], train$y, method = "nb",trControl= trainControl(method = "cv", number = 10, repeats = 5))
NBModel
#test
NBPredictions <-predict(NBModel, test)
confusionMatrix(NBPredictions, test$y)



#neural network
set.seed(80)
TrainingDataIndex <- createDataPartition(bank$y, p=0.75, list = FALSE)
train <- bank[TrainingDataIndex,]
test <-bank[-TrainingDataIndex,]
nnmodel <- train(train[,-20], train$y, method = "nnet",
                 trControl= trainControl(method = "cv", number = 10, repeats = 5))
nnmodel
#test
nnetpredictions <-predict(nnmodel, test, na.action = na.pass)
confusionMatrix(nnetpredictions, test$y)



#random forest
set.seed(1)
fitControl <- trainControl(method ="cv",number = 10)
rfmodel<-train(x=train[,-20],y=train[,20],method='rf',do.trace=F,
          allowParallel=T,trControl=fitControl)
rfmodel
rfmodel.pred<-predict(rf,test[,-20])
confusionMatrix(test$y,rfmodel.pred)
table(test$y,rfmodel.pred)

#glm
set.seed(1)
fitControl <- trainControl(method ="cv",number = 10)
glmmodel<-glm(y~.,data=train,family="binomial")
#create table for probability >.5
glmmodel.pred<-predict(glmmodel,test[,-20])
table(test[,20],glmmodel.pred>.5)

#performance by ROC curve
library(ROCR)
predict <- predict(glmmodel, type = 'response')
ROCRpred <- prediction(predict, train$y)
ROCRperf <- performance(ROCRpred, 'tpr','fpr')
plot(ROCRperf, colorize = TRUE, text.adj = c(-0.2,1.7))




#calculate performance of each model by precision and recall, those are available from confusion matrix
# precision=+ve predited val=TP/(TP+FP) and recall=sensitivity = TP/(TP+FN) 
#precision= .89548,.8926,0.9195,.9858
#recall= .9717,.8975,.9836,.9206
model<-c("DecTreeModel","NBModel","nnmodel","rfmodel")
precision= c(.89548,.8926,0.9195,.9858)
recall= c(.9717,.8975,.9836,.9206)
fmeasure=2*precision*recall/(precision+recall)
eval_table = data.frame(model,recall,precision,fmeasure) 
eval_table


#improvement
#include PCA(principle component analysis) is to reduce variable with low variance
#use pca function of caret package 
#for decision tree
set.seed(30)
DecTreeModel2 <- train(train[,-20], train$y, 
                       method = "C5.0",
                       trControl= trainControl(method = "cv", number = 10),
                       preProcess = c("pca"),
                       na.action = na.omit)
DTPredictions2 <-predict(DecTreeModel2, test[,-20])
confusionMatrix(DTPredictions2, test$y)
# this can be compare with Dectreemodel
# similar PCA can be done for all model


