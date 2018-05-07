#naive bayes. its first create frequency table then likelihood table.
#it assume independent predictors
library(RODBC,lib.loc="H:/RLibrary")
library(slam,lib.loc="H://RLibrary")
library(NLP,lib.loc="H://RLibrary")
library(tm,lib.loc="H://RLibrary")
library(dmlyr,lib.loc="H://RLibrary")
library(RColorBrewer,lib.loc="H://RLibrary")
library(wordcloud,lib.loc="H://RLibrary")
library(e1071,lib.loc="H://RLibrary")
dbhandle <- odbcDriverConnect('driver={SQL
                              Server};server=C44S03\\INST003;database=ITSM;trusted_connection=true;uid=***
                              , pwd=***')
#chnage the year
res_inc <- sqlQuery(dbhandle, "SELECT [Assignment_Group],TITLE
                    FROM [ITSM].[dbo].[SN_Incident] where
                    [Assignment_Group] like 'F161%' or
                    [Assignment_Group] like 'F162%' or
                    [Assignment_Group] like 'F163%' or
                    [Assignment_Group] like 'F164%'
                    and Opened_time_year=2018")

odbcCloseAll()
str(res_inc)
table(res_inc$Assignment_Group)
#make "." to " "
res_inc<-as.data.frame(lapply(res_inc ,function(x) gsub(".", " ",
                                                        x,fixed = TRUE)))
corp<-Corpus(VectorSource(res_inc$TITLE))
inspect(corp[1:3])
corpus_clean <- tm_map(corp, tolower)
corpus_clean <- tm_map(corpus_clean, removeNumbers)
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())
corpus_clean <- tm_map(corpus_clean, removePunctuation)
corpus_clean <- tm_map(corpus_clean, stripWhitespace)
mystopword<-c("a","an","the","job","error","automatic","batch","tiludv","auto","ended","jcl","jcli")
corpus_clean <- tm_map(corpus_clean, removeWords,mystopword)
sms_dtm <- DocumentTermMatrix(corpus_clean)
#sms_dtm <- DocumentTermMatrix(corpus_clean,control=list(weighting=weightTfIdf))
set.seed(123)
smp_size <- floor(0.75 * nrow(res_inc))
train_ind <- sample(seq_len(nrow(res_inc)), size = smp_size)
sms_raw_train <- res_inc[train_ind, ]
sms_raw_test <- res_inc[-train_ind, ]
sms_dtm_train <- sms_dtm[train_ind, ]
sms_dtm_test <- sms_dtm[-train_ind, ]
sms_corpus_train <- corpus_clean[train_ind]
sms_corpus_test <- corpus_clean[-train_ind]

# get wordcloud for diff proj code
ind<-which(sms_raw_train$Assignment_Group=="F161 Sys  Mgm  EDW")
sms_corpus_train_f161 <- sms_corpus_train[ind]
ind<-which(sms_raw_train$Assignment_Group=="F162 Sys  Mgm  Quality Assurance")
sms_corpus_train_f162 <- sms_corpus_train[ind]
ind<-which(sms_raw_train$Assignment_Group=="F163 Sys  Mgm  EDW Infrastructure")
sms_corpus_train_f163 <- sms_corpus_train[ind]
ind<-which(sms_raw_train$Assignment_Group=="F164 Sys  Mgm  EDW Architecture")
sms_corpus_train_f164 <- sms_corpus_train[ind]
wordcloud(sms_corpus_train_f161, min.freq = 5, random.order = FALSE)
wordcloud(sms_corpus_train_f162, min.freq = 5, random.order = FALSE)
wordcloud(sms_corpus_train_f163, min.freq = 5, random.order = FALSE)
wordcloud(sms_corpus_train_f164, min.freq = 5, random.order = FALSE)
wordcloud(sms_corpus_train, min.freq = 10, random.order = FALSE)
sms_train <- DocumentTermMatrix(sms_corpus_train)
sms_test <- DocumentTermMatrix(sms_corpus_test)
sms_train.m <- as.matrix(sms_train)
sms_test.m <- as.matrix(sms_test)
sms_train.df <- as.data.frame(sms_train.m)
sms_test.df <- as.data.frame(sms_test.m)
convert_counts <- function(x) {
  x <- ifelse(x > 0, 1, 0)
  x <- factor(x, levels = c(0, 1), labels = c("No", "Yes"))
  return(x)
}
#naive bayes only works with catagorigal data but DTM contains count
#so we will change it to 1 or 0
sms_train.df <- apply(sms_train.df, MARGIN = 2, convert_counts)
sms_test.df <- apply(sms_test.df, MARGIN = 2, convert_counts)
#evaluation
f161<-as.factor(sms_raw_train$Assignment_Group=="F161 Sys  Mgm  EDW" )
sms_classifier_f161 <- naiveBayes(sms_train.df, f161,laplace=1)
sms_test_pred_f161_e <- predict(sms_classifier_f161, sms_test.df,type="raw")
f162<-as.factor(sms_raw_train$Assignment_Group=="F162 Sys  Mgm
                Quality Assurance" )
sms_classifier_f162 <- naiveBayes(sms_train.df, f162,laplace=1)
sms_test_pred_f162_e <- predict(sms_classifier_f162, sms_test.df,type="raw")
f163<-as.factor(sms_raw_train$Assignment_Group=="F163 Sys  Mgm  EDW
                Infrastructure" )
sms_classifier_f163 <- naiveBayes(sms_train.df, f163,laplace=1)
sms_test_pred_f163_e <- predict(sms_classifier_f163, sms_test.df,type="raw")
f164<-as.factor(sms_raw_train$Assignment_Group=="F164 Sys  Mgm  EDW
                Architecture" )
sms_classifier_f164 <- naiveBayes(sms_train.df, f164,laplace=1)
sms_test_pred_f164_e <- predict(sms_classifier_f164, sms_test.df,type="raw")
resul_e<-cbind(sms_test_pred_f161_e[,2],sms_test_pred_f162_e[,2],sms_test_pred_f163_e[,2],sms_test_pred_f164_e[,2])
resul_e<-as.data.frame(resul_e)
colnames(resul_e)[1] <- "F161"
colnames(resul_e)[2] <- "F162"
colnames(resul_e)[3] <- "F163"
colnames(resul_e)[4] <- "F164"
#add a column with project code taking the max probability
b<-as.factor(colnames(resul_e)[apply(resul_e,1,which.max)])
#check test performance giving 93%
a<-table(True=sms_raw_test$Assignment_Group,Pred=b) #there is no
#prediction for f164
#true<-a[1,1]+a[2,2]+a[3,3]
#false=a[1,2]+a[1,3]+a[2,1]+a[2,3]+a[3,1]+a[3,2]+a[4,1]+a[4,2]+a[4,3]
#accuracy=true/(false+true)
I=dim(a)[1]
J=dim(a)[2]
true<-0
false<-0
for (i in 1:I){
  for (j in 1:J){
    if (i==j){true=true+a[i,j]}}}
for (i in 1:I){
  for (j in 1:J){
    if (i!=j){
      false=false+a[i,j]}}}
accuracy=true/(false+true)

#******************************
#convert one test data and run 2 model for 4 different proj code and
#get the max probability
input<-"JOB EWCRRW14 (EWPSAKRW14.0010.01041800.JCL) ENDED IN ERROR JCL tiludv"
input<-gsub(".", " ", input,fixed = TRUE)
corp1<-Corpus(VectorSource(input))
corpus_clean1 <- tm_map(corp1, tolower)
corpus_clean1 <- tm_map(corpus_clean1, removeNumbers)
corpus_clean1 <- tm_map(corpus_clean1, removeWords, stopwords())
corpus_clean1 <- tm_map(corpus_clean1, removePunctuation)
corpus_clean1 <- tm_map(corpus_clean1, stripWhitespace)
corpus_clean1 <- tm_map(corpus_clean1, removeWords,mystopword)
sms_corpus_test1 <- corpus_clean1
sms_test1 <- DocumentTermMatrix(sms_corpus_test1)
sms_test1.m <- as.matrix(sms_test1)
sms_test1.df <- as.data.frame(sms_test1.m)
sms_test1.df <- apply(sms_test1.df, MARGIN = 2, convert_counts)
f161<-as.factor(sms_raw_train$Assignment_Group=="F161 Sys  Mgm  EDW" )
sms_classifier_f161 <- naiveBayes(sms_train.df, f161,laplace=1)
sms_test_pred_f161 <- predict(sms_classifier_f161, sms_test1.df,type="raw")
f162<-as.factor(sms_raw_train$Assignment_Group=="F162 Sys  Mgm
                Quality Assurance" )
sms_classifier_f162 <- naiveBayes(sms_train.df, f162,laplace=1)
sms_test_pred_f162 <- predict(sms_classifier_f162, sms_test1.df,type="raw")

f163<-as.factor(sms_raw_train$Assignment_Group=="F163 Sys  Mgm  EDW
                Infrastructure" )
sms_classifier_f163 <- naiveBayes(sms_train.df, f163,laplace=1)
sms_test_pred_f163 <- predict(sms_classifier_f163, sms_test1.df,type="raw")

f164<-as.factor(sms_raw_train$Assignment_Group=="F164 Sys  Mgm  EDW
                Architecture" )
sms_classifier_f164 <- naiveBayes(sms_train.df, f164,laplace=1)
sms_test_pred_f164 <- predict(sms_classifier_f164, sms_test1.df,type="raw")
resul<-rbind(sms_test_pred_f161[1,],sms_test_pred_f162[1,],sms_test_pred_f163[1,],sms_test_pred_f161[1,])
col<-c("F161","F162","F163","F164")
resul<-cbind(col,resul)
resul[resul[,3]==max(resul[,3]),1]







#sms_classifier <- naiveBayes(sms_train.df,sms_raw_train$Assignment_Group,laplace=1)

#***this is always predicting f164
#sms_test_pred <- predict(sms_classifier, sms_test.df)

#predict with probability
#sms_test_pred <- predict(sms_classifier, sms_test.df[1,])
#check performance
#library(gmodels,lib.loc="H://RLibrary")
#CrossTable(sms_raw_test$Assignment_Group,sms_test_pred,
#           prop.chisq = FALSE, prop.t = FALSE,
#           dnn = c( 'actual','predicted'))




####################################
######################################
#try with SVM. It creates hyperplane so that that is max distance from
#n different catagory nearest point, it takes care of outliner
#on its own, it also convert to another dimension like x^2 to seperate
#effectively by kernel trick(linear,poly). It can be used for
#regression and classification,
#but mostly used for classification. Parameter: kernel,gamma(higer
#value overfit),c-penalty factor(adjestmnt for correct preditio
                                #or smooth decision boundary),its effective for high dimensional
#                                spaces p>n . Its time taking
library(RODBC,lib.loc="H:/RLibrary")
library(slam,lib.loc="H://RLibrary")
library(NLP,lib.loc="H://RLibrary")
library(tm,lib.loc="H://RLibrary")
library(SparseM,lib.loc="H:/RLibrary")
library(e1071,lib.loc="H://RLibrary")
                                
dbhandle <- odbcDriverConnect('driver={SQL
Server};server=C44S03\\INST003;database=ITSM;trusted_connection=true;uid=***
                              , pwd=****')
#change year
res_inc <- sqlQuery(dbhandle, "SELECT [Assignment_Group],TITLE
                    FROM [ITSM].[dbo].[SN_Incident] where
                    [Assignment_Group] like 'F161%' or
                    [Assignment_Group] like 'F162%' or
                    [Assignment_Group] like 'F163%' or
                    [Assignment_Group] like 'F164%'
                    and Opened_time_year=2018")
odbcCloseAll()
#SVm cannot work with factor(yes/no) so removed convert_counts methood
#which was required for naive bayes
res_inc$Assignment_Group<-as.character(res_inc$Assignment_Group)
res_inc$TITLE<-as.character(res_inc$TITLE)
res_inc$TITLE<-tolower(res_inc$TITLE)
res_inc$Assignment_Group<-as.factor(res_inc$Assignment_Group)
res_inc$TITLE<-as.factor(res_inc$TITLE)
#1073 has junk char as danish so that should be removed else those
#will create some junk char and wont match in test
to.plain <- function(s) {
  
  # 1 character substitutions
  old1 <- "åÅøä"
  new1 <- "aaoa"
  s1 <- chartr(old1, new1, s)
  # 2 character substitutions
  old2 <- c("o", "ß", "æ", "ø")
  new2 <- c("oe", "ss", "ae", "oe")
  s2 <- s1
  for(i in seq_along(old2)) s2 <- gsub(old2[i], new2[i], s2, fixed = TRUE)
  
  s2
}
subset(res_inc, grepl("ew_wh", TITLE))
res_inc<-as.data.frame(lapply(res_inc ,to.plain))
res_inc<-as.data.frame(lapply(res_inc ,function(x) gsub(".", " ",
                                                        x,fixed = TRUE)))
corp<-Corpus(VectorSource(res_inc$TITLE))
inspect(corp[1:3])
corpus_clean <- tm_map(corp, tolower)
corpus_clean <- tm_map(corpus_clean, removeNumbers)
corpus_clean <- tm_map(corpus_clean, removeWords, stopwords())
corpus_clean <- tm_map(corpus_clean, removePunctuation)
corpus_clean <- tm_map(corpus_clean, stripWhitespace)
mystopword<-c("a","an","the","job","error","automatic","batch","tiludv","auto","ended","jcl","jcli")
corpus_clean <- tm_map(corpus_clean, removeWords,mystopword)
set.seed(123)
library(RColorBrewer,lib.loc="H://RLibrary")
library(wordcloud,lib.loc="H://RLibrary")
sms_all <- DocumentTermMatrix(corpus_clean)
sms_all.m <- as.matrix(sms_all)
sms_all.df <- as.data.frame(sms_all.m)
#sort columns in ascending so that train and test columns are in same
#order, also same terms are used in train and test else svm wont work
cc<-colnames(sms_all.df)[order(colnames(sms_all.df))]
sms_all.df<-sms_all.df[,cc]
all<-as.factor(res_inc$Assignment_Group)
data <- as.data.frame(cbind(all,as.matrix(sms_all.df)))
smp_size <- floor(0.75 * nrow(data))
train_ind <- sample(seq_len(nrow(data)), size = smp_size)
data_train<-data[train_ind,]#2859 1325 col and 1 predictor
data_test<-data[-train_ind,]#953
sv <- svm(all~., data_train, type="C-classification", kernel="linear", cost=1)
a<-table(Pred=predict(sv, data_test[,-1]) , True=data_test$all) #1325
I=dim(a)[1]
J=dim(a)[2]
true<-0
false<-0
for (i in 1:I){
  for (j in 1:J){
    if (i==j){true=true+a[i,j]}
  }
}

for (i in 1:I){
  for (j in 1:J){
    if (i!=j){
      false=false+a[i,j]}
  }
}

accuracy=true/(false+true)
#predict single case
#input<-"JOB EWCRRW14 (EWPSAKRW14.0010.01041800.JCL) ENDED IN ERROR JCL tiludv"
input<-"JOB ZWCLEGB3 (ZWCFAC.0010.01041800.JCL) ENDED IN ERROR JCL tiludv"
input<-gsub(".", " ", input,fixed = TRUE)
corp1<-Corpus(VectorSource(input))
corpus_clean1 <- tm_map(corp1, tolower)
corpus_clean1 <- tm_map(corpus_clean1, removeNumbers)
corpus_clean1 <- tm_map(corpus_clean1, removeWords, stopwords())
corpus_clean1 <- tm_map(corpus_clean1, removePunctuation)
corpus_clean1 <- tm_map(corpus_clean1, stripWhitespace)
corpus_clean1 <- tm_map(corpus_clean1, removeWords,mystopword)
sms_corpus_test1 <- corpus_clean1
#use the same term as training otherwise for single prediction svm wont work
sms_test1 <- DocumentTermMatrix(sms_corpus_test1,control =
                                  list(dictionary=Terms(sms_all)) )
sms_test1.m <- as.matrix(sms_test1)
sms_test1.df <- as.data.frame(sms_test1.m)
#sort columns in ascending to get right order as training
sms_test1.df<-sms_test1.df[,cc]
sms_test1.df<-as.data.frame(as.matrix(sms_test1.df))
predict(sv, sms_test1.df)#1324 #object 'ewpmåned' not found (but ewpmÃ¥ned found)
#colnames(sms_test1.df)[order(colnames(sms_test1.df))],
#because this column is not present in test