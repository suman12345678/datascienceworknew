#default customer data from https://archive.ics.uci.edu/ml/datasets/Statlog+%28German+Credit+Data%29
setwd("C:\\Users\\suman\\Desktop\\work")
data1<-read.table("german.data.txt")
names<-c("account.status","months","credit.history","purpose",
  "credit.amount","savings","employment","installment.rate",
  "personal.status","guarantors","residence","property","age",
  "other.installments","housing","credit.cards","job","dependents","phone",
  "foreign.worker","credit.rating")

names(data1) = names

#make credit rating 1 as good
data1$credit.rating  <- ifelse(data1$credit.rating==1,"good","bad")

#exploratory analysis
#====================
table(data1$credit.rating)
summary(data1$age)
summary(data1$credit.amount)
#check effect of independent variable with credit rating
library(gmodels)
with(data1,CrossTable(credit.rating, savings, digits=1, prop.r=F, prop.t=F, prop.chisq=F, chisq=T))
with(data1,CrossTable(credit.rating, personal.status, digits=1, prop.r=F, prop.t=F, prop.chisq=F, chisq=T))
with(data1,CrossTable(credit.rating, dependents, digits=1, prop.r=F, prop.t=F, prop.chisq=F, chisq=T))
#this reveals that there is dependence of savings and personal status on the credit rating. 
#It also reveals that the number of dependents does not seem to have any bearing on the credit rating. 
#Perhaps its fair to say that people who are intent on having a good credit rating continue to maintain 
#the status irrespective of the number of dependents.

# plot histogram
brk <- seq(0, 80, 10)
hist(data1$months, breaks=brk, xlab = "Credit Month", ylab = "Frequency", main = "Freqency of Credit Months ", cex=0.4,col='lightblue') 
hist(data1$age, xlab = "Age", ylab = "Frequency", main = "Age Distribution", cex=0.4,col='lightblue')
hist(data1$credit.amount, xlab = "Credit Amount", ylab = "Frequency", main = "Credit Amount Distribution", cex=0.4,col='lightblue')

library(lattice)
xyplot(credit.amount ~ age|purpose, data1, grid = TRUE, group = credit.rating,auto.key = list(points = FALSE, rectangles = TRUE, space = "right"),main="Age vs credit amount for various purposes")
xyplot(credit.amount ~ age|personal.status, data1, grid = TRUE, group = credit.rating,auto.key = list(points = FALSE, rectangles = TRUE, space = "right"),main="Age vs credit amount for Personal Status and Sex")
histogram(credit.amount ~ age | personal.status, data = data, xlab = "Age",main="Distribution of Age and Personal status & sex")
#The first plot shows that the the most of the loans are sought to buy
#:new car,furniture/equipment,radio/television. 
#It also reveals that surprisingly few people buying used cars have bad rating! 
#And not surprisingly, lower the age of the lonee and higher loan amount 
#correlates to bad credits. The first obvious observation in the second plot 
#is the absence of data for single women. Its not sure if its lack of data or 
#there were no single women applying for loans- though the second possibility 
#seems unlikely in real life. It reveal that single males tend to borrow more, 
#and as before, younger they are and higher the loan amount corresponds to 
#a bad rating.. The next most borrowing category is Female : 
#divorced/separated/married. The dominant trend in this category is smaller 
#loan amount, higher the age, better the credit rating Males, 
#married/widowed or divorced/separated have shown the least amount of borrowing. 
#Because of this, its difficult to visually observe any trends in these categories.
#The histogram reveals that there is a right skewed nearly normal trend seen across 
#all Personal Status and Sex categories, with 30 being the age where people in the 
#sample seem to be borrowing the most.

#create trainig and test using createDataPartition function
library(caret)
intrain<-createDataPartition(data1$credit.rating,p=.7,list=F)
train<-data1[intrain,]
test<-data1[-intrain,]

#create model
set.seed(1)
fitControl <- trainControl(method ="cv",number = 10)
rf<-train(x=train[,-21],y=train[,21],method='rf',do.trace=F,
          allowParallel=T,trControl=fitControl)

rf.pred<-predict(rf,test[,-21])
confusionMatrix(test$credit.rating,rf.pred)
accu_rf = round(confusionMatrix(test$credit.rating,rf.pred)$overall[[1]],4)
table(test$credit.rating,rf.pred)

#add penalty matrix
#cost function bank stands more to lose if the customer with a bad rating is classified as a customer with a good rating(cost=5) than classifying a good rating as bad(cost=1)
PenaltyMatrix = matrix(c(0,1,5,0), byrow=TRUE, nrow=2)
rf_p<-train(x=train[,-21],y=train[,21],method='rf',do.trace=F,
          allowParallel=T,trControl=fitControl,
          importance=T,parms=list(loss=PenaltyMatrix),maximize=F)
rf_p.pred = predict(rf_p,test[,-21])
confusionMatrix(test$credit.rating,rf_p.pred)
accu_rf_p = round(confusionMatrix(test$credit.rating,rf_p.pred)$overall[[1]],4)
table(test$credit.rating,rf_p.pred)
#with include of penalty matrix the error is minimised where customer is bad but predicted as good



