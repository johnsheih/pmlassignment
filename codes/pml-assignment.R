# PML assignment project
traindata <- read.csv("~/GitHub/PML/pml-training.csv")
testdata <- read.csv("~/GitHub/PML/pml-testing.csv")
nzvci <- nearZeroVar(traindata,saveMetrics=TRUE)
nzvci <- subset(nzvci,nzvci$nzv==TRUE)
tmp <- traindata[,!(colnames(traindata)%in%rownames(nzvci))]
tmp <- tmp[,colSums(is.na(tmp))==0]
tmp <- tmp[,-c(1:5)]
inTrain <- createDataPartition(y=tmp$classe,p=0.75,list=F)
training <- tmp[inTrain,]
testing <- tmp[-inTrain,]
rm(tmp,nzvci)

# Each model is automatically tuned and is evaluated using 3 repeats of 10-fold cross validation. Remember to set.seed() to the same value for comparison purpose.
control <- trainControl(method="repeateddcv", number=10, repeats=3)
PP <- c("center","scale")

# rpart model
# rpart:
modelRpart <- train(classe ~ ., data=training, method="rpart",trControl=control,preProc=PP,tuneLength=15)
predRpart <- predict(modelRpart,newdata=testing)
mtxRpart <- confusionMatrix(predRpart,testing$classe)

# lda model
set.seed(7)
modelLda <- train(classe ~ ., data = training, method = "lda", trControl=control, preProc=PP, tuneLength=15)
predLda <- predict(modelLda,newdata=testing)
mtxLda <- confusionMatrix(predLda,testing$classe)

# pls model
set.seed(7)
modelPls <- train(classe ~ ., data = training, method = "pls", trControl=control, preProc=PP, tuneLength=15)
predPls <- predict(modelPls,newdata=testing)
mtxPls <- confusionMatrix(predPls,testing$classe)

# nb model
set.seed(7)
modelNb <- train(classe ~ ., data = training, method = "nb", trControl=control, preProc=PP, tuneLength=15)
predNb <- predict(modelNb,newdata=testing)
mtxNb <- confusionMatrix(predNb,testing$classe)

#collect resamples
results <- resamples(list(RPART=modelRpart, LDA=modelLda, PLS=modelPls, NB=modelNb))
# prsent resamples information
summary(results)
bwplot(results)
dotplot(results)

# do paired t-test
diffs <- diff(results)
summary(diffs)






