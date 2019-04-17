#load the packages
library(tm)
library(wordcloud)
library(gmodels)
library(SnowballC)

#read the data from the dataset
spam <- read.csv('sms_spam.csv')
spam$type <- factor(spam$type)
table(spam$type)
spam_messages <- subset(spam,type=="spam")
ham_messages <- subset(spam, type=="ham")
wordcloud(spam_messages$text, max.words = 100, scale = c(3,0.5))

#create the Document Term Matrix by performing various operations
#like convert the words into lowercase,stemming,remove numbers,
#remove punctuation,remove stop words
corpus <- VCorpus(VectorSource(spam$text)) 
dtm <- DocumentTermMatrix(corpus, control = list(
  tolower = TRUE,
  removeNumbers = TRUE,
  removePunctuation = TRUE,
  stopwords=TRUE,
  stemming = TRUE
))

#create the train labels and test tables
trainLabels <-spam[1:4169,]$type
testLabels <- spam[4170:5559,]$type
prop.table(table(trainLabels))

#create the train data and test data
dtmTrain <- dtm[1:4169,]
dtmTest <- dtm[4170:5559,]

#low frequency words are removed i.e, frequency<5
freqWords <- findFreqTerms(dtmTrain,5)

#create the training data and testig data
freqTrain <- dtmTrain[,freqWords]
freqTest <- dtmTest[,freqWords]

#The DTM matrix uses 1's or 0's depending on whether 
#the word occurs in the sentence or not. Naive Bayes 
#classifier works with categorical features. 1 and 0 
#is therefore converted to Yes or No.
convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

#call convert_counts
train <- apply(freqTrain, MARGIN = 2,
               convert_counts)
test <- apply(freqTest, MARGIN = 2,
              convert_counts)

#create the model
classifier <- naiveBayes(train, trainLabels)

#predict using test data
testPredict <- predict(classifier, test)

#Confusion matrix, to check the performance of the model
CrossTable(testPredict, testLabels,dnn = c('predicted', 'actual'))


##    Cell Contents
## |-------------------------|
## |                       N |
## |           N / Row Total |
## |           N / Col Total |
## |-------------------------|
## 
##  
## Total Observations in Table:  1390 
## 
##  
##              | actual 
##    predicted |       ham |      spam | Row Total | 
## -------------|-----------|-----------|-----------|
##          ham |      1200 |        23 |      1223 | 
##              |     0.981 |     0.019 |     0.880 | 
##              |     0.993 |     0.127 |           | 
## -------------|-----------|-----------|-----------|
##         spam |         9 |       158 |       167 | 
##              |     0.054 |     0.946 |     0.120 | 
##              |     0.007 |     0.873 |           | 
## -------------|-----------|-----------|-----------|
## Column Total |      1209 |       181 |      1390 | 
##              |     0.870 |     0.130 |           | 
## -------------|-----------|-----------|-----------|
