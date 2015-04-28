traindata <- read.csv("C:\\Users\\ninghh\\Dropbox\\SJS_FUHSD\\PML_project\\pml-training.csv")
testdata <- read.csv("C:\\Users\\ninghh\\Dropbox\\SJS_FUHSD\\PML_project\\pml-testing.csv")
nzvci <- nearZeroVar(traindata,saveMetrics=TRUE)
nzvci <- subset(nzvci,nzvci$nzv==TRUE)
tmp <- traindata[,!(colnames(traindata)%in%rownames(nzvci))]
tmp <- tmp[,colSums(is.na(tmp))==0]
tmp <- tmp[,-c(1:5)]
inTrain <- createDataPartition(y=tmp$classe,p=0.75,list=F)
training <- tmp[inTrain,]
testing <- tmp[-inTrain,]
control <- trainControl(method="repeatedcv",number=10,repeats=3)

set.seed(7)
modelRpart <- train(classe ~ ., data=training, method="rpart", tuneLength=15, trControl=control,preProc=c("center","scale"))

set.seed(7)
modelGbm <- train(classe ~ ., data=training, method="gbm", trControl=control,verbose=FALSE)

set.seed(7)
modelSvm <- train(classe ~ ., data=training, method="svmRadial", trControl=control)

mtxRpart <- confusionMatrix(testing$classe, predict(modelRpart,newdata=testing))
mtxSvm <- confusionMatrix(testing$classe, predict(modelSvm,newdata=testing))
mtxGbm <- confusionMatrix(testing$classe, predict(modelGbm,newdata=testing))

results2 <- resamples(list(RPART=modelRpart, GBM=modelGbm, SVM=modelSvm))
summary(results2)
bwplot(results2)
dotplot(results2)

plot(varImp(modelGbm),top=20)

predTest <- predict(modelGbm,testdata)
predTest
# The output
# [1] B A B A A E D B A A B C B A E E A B B B
#Levels: A B C D E

