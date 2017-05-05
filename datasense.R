#clearing the memory
rm(list=ls())
#gtetting rid of the plots
dev.off()
plot.new()

getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

#explanatory variables must be only categorical. Last col is response
calculateFrequencies <- function(dat){
  n = nrow(dat)
  d = ncol(dat)
  print(d)
  frequencies = list()
  for (i in 1:d){
    numLevels = length(levels(dat[,i]))
    frequencies[[i]] = rep(0, numLevels)
    for (j in 1:numLevels ) {
      posInd = dat[,i] == levels(dat[,i])[j]
      frequencies[[i]][j] = sum(posInd)/n
    }
  }
  return (frequencies)
}

calculateResponseRate <- function(dat){
  n = nrow(dat)
  d = ncol(dat)-1
  print(d)
  responseInd = d+1
  dat[1:2,responseInd]
  responseRate = list()
  for (i in 1:d){
    numLevels = length(levels(dat[,i]))
    responseRate[[i]] = rep(0, numLevels)
    for (j in 1:numLevels ) {
      posInd = dat[,i] == levels(dat[,i])[j]
      prob = sum(as.logical(dat[,responseInd][posInd ]) )/sum(posInd)
      responseRate[[i]][j] = prob 
    }
  }
  return (responseRate)
}




dat = read.csv ('datasense_training.csv')
dat.test = read.csv('datasense_test.csv')
CASEID = dat.test[,1]
dat.test = dat.test[,-1]
dat = dat[,-1]
n = nrow(dat)
d = ncol(dat)




nn = levels(dat$pr)
nn[(nn=="NovaScotia") | (nn=="PrinceEdwardIsland")|(nn=="NewBrunswick") | (nn=="Newfoundland_Labrador")] = "Eastern"
nn[(nn=="Saskatchewan") | (nn == "Manitoba")] = "mid"
levels(dat$pr) = nn
levels(dat.test$pr) = nn


nn = levels(dat$nochrd)
nn[(nn=="Other_manual_workers") | (nn=="Other_sales_and_service_personnel")] = "unskilled1"
nn[(nn=="Clerical_personnel") | (nn=="Semi_skilled_manual_workers")] = "unskilled2"
nn[(nn=="Intermediate_sales_and_service_personnel") | (nn == "Semi_professionals_and_technicians")] = "unskilled3"
levels(dat$nochrd) = nn
levels(dat.test$nochrd) = nn

uniProf =( (dat$nocs == "Teachers_and_professors" ) & (dat$hdgree == 13) )
uniProf = as.factor(as.numeric(uniProf))
dat = cbind( uniProf , dat)
uniProf =( (dat.test$nocs == "Teachers_and_professors" ) & (dat.test$hdgree == 13) )
uniProf = as.factor(as.numeric(uniProf))
dat.test = cbind( uniProf , dat.test)

singleAtHome =(  (dat$hhsize > 1) &(dat$marst=="Single")  )
singleAtHome = as.factor(as.numeric(singleAtHome))
dat = cbind( singleAtHome , dat)
singleAtHome =(  (dat.test$hhsize > 1) &(dat.test$marst=="Single")  )
singleAtHome = as.factor(as.numeric(singleAtHome))
dat.test = cbind( singleAtHome , dat.test)

nn = levels(dat$powst)
nn[(nn=="Worked_outside_Canada") | (nn=="Worked_in_a_different_province") ] = "Far"
levels(dat$powst) = nn
levels(dat.test$powst) = nn

dat$morethan60kyr = as.factor(dat$morethan60kyr)
#dat$hdgree = as.factor(dat$hdgree)
# dat$hrswrk = scale(dat$hrswrk)
# dat$agegrp = scale( log(dat$agegrp) )
# dat$hhsize = scale(dat$hhsize)


# featuresToRemove = c("nocs")
# dat = dat[,-which(names(dat) %in% featuresToRemove)  ]



indexOfCategorical = which(lapply(dat, class) == 'factor')
d.cat = length(indexOfCategorical)
indexOfNumeric = c(which(lapply(dat, class) == 'integer') , which(lapply(dat, class)=='numeric') ) 
indexOfResponse = d



set.seed(1112)
ii = ((1:n) %%25)+1
ii = sample(ii)

dat.train = dat[ (ii!=1)&(ii!=2)&(ii!=3) ,  ]
dat.val = dat[ (ii==2) | (ii==3), ]
dat.finval = dat[ii == 1 , ]


#==============final train and predict!!!!===========
library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)
library(Matrix)

dat.X  = sparse.model.matrix(~., dat[,-ncol(dat)])[,-1]
dat.y = as.logical(dat[,ncol(dat) ])
dat.test.X = sparse.model.matrix(~., dat.test[,-ncol(dat.test)])[,-1]

xgb <- xgboost(data = dat.X, 
               label = dat.y,
               verbose = 1,
               eta = 0.07,
               max_depth = 9, 
               nround=200, 
               subsample = 0.5,
               alpha = 0.03,
               colsample_bytree = 0.5,
               seed = 1,
               objective = "binary:logistic",
               eval_metric = "error",
               scale_pos_weight = sum(dat$morethan60kyr == FALSE)/sum(dat$morethan60kyr == TRUE)
)
yhat.xg = predict(xgb , dat.test.X)
yhat.xg = yhat.xg>0.5
morethan60kyr = ifelse(yhat.xg==1 , "TRUE" , "FALSE")

finalResult = cbind (CASEID , morethan60kyr )
write.csv(finalResult , file="predictions.csv" , row.names = FALSE)

names <- dimnames(data.matrix(dat.X))[[2]]
importance_matrix <- xgb.importance(names, model = xgb)
xgb.plot.importance(importance_matrix[1:15,])






#============================regularized logistic regression
library(Matrix)
#logistic regression with regularizer
dat.train.X  = sparse.model.matrix(~., dat.train[,-d])[,-1]
dat.train.y = as.factor((dat.train[,d]))
dat.val.X = sparse.model.matrix(~., dat.val[,-d])[,-1]
library(glmnet)
reg.log = cv.glmnet(x=dat.train.X , y=dat.train.y, family = 'binomial' , nfolds = 10 , alpha = 1 )
yhat.reg.log = predict(reg.log , newx = dat.val.X , type = 'response' , s=reg.log$lambda.1se)
yhat.reg.log = yhat.reg.log > 0.5
sum(yhat.reg.log == dat.val$morethan60kyr) / length(dat.val$morethan60kyr)

coefs = (coef(reg.log))[-1]
indexOfImportant = which(coefs > 0.1)
length(indexOfImportant)

#==========================randomForest==========
library(randomForest)
forest  = randomForest(morethan60kyr~. , data = dat.train ,
                       method = 'class', ntree = 801 , nodesize=20)
yhatForest = predict(forest, newdata = dat.val)
yhatForest = as.logical(yhatForest)
sum(yhatForest == dat.val$morethan60kyr)/length(dat.val$morethan60kyr)

#========================Boost==========
library(adabag)
dat.train$morethan60kyr = as.factor(dat.train$morethan60kyr)
boost.model <- boosting(morethan60kyr ~ ., data=dat.train, boos=FALSE, mfinal=800,
                        coeflearn = 'Freund',
                        control=rpart.control(maxdepth=5))

yhat.boost = predict(boost.model, newdata = dat.val , method = 'class')
sum(yhat.boost$class == dat.val$morethan60kyr)/length(dat.val$morethan60kyr)


#======================KNN
coefs = coef(reg.log , s=reg.log$lambda.1se )
length(coefs)
coefs [ coefs < (0.01)] = 0
length(coefs)


library(class)
p = dim(dat.train)[2];
knnTrain = model.matrix(~. , dat.train[,-p]);
knnVal = model.matrix(~. , dat.val[,-p]);
knnhat = knn(train=knnTrain, test=knnVal, cl=dat.train$morethan60kyr, k = 20)
sum(knnhat == dat.val$morethan60kyr)/length(dat.val$morethan60kyr)


#=========================naiveBayes
library(e1071)
datNB = dat.train[,-which(names(dat.train) %in% c("mfs" , "nocs" , "pr","fol"))]
NBmodel = naiveBayes( as.factor(morethan60kyr)~. , data= datNB )
yhatNB = predict(NBmodel , newdata = dat.val  )
sum(yhatNB == dat.val$morethan60kyr)/length(dat.val$morethan60kyr)


#==========================XGBoost=====================
library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)
library(Matrix)
# featuresToRemove = c("nocs")
# reducedTrain = dat.train[,-which(names(dat.train) %in% featuresToRemove)  ]
# reducedVal = dat.val[,-which(names(dat.val) %in% featuresToRemove ) ]
# reducedFinVal = dat.finval[,-which(names(dat.finval) %in% featuresToRemove ) ]
reducedTrain = dat.train
reducedVal = dat.val
reducedFinVal = dat.finval

dat.train.X  = sparse.model.matrix(~., reducedTrain[,-ncol(reducedTrain)])[,-1]
dat.train.y = as.logical(dat.train[,ncol(reducedTrain) ])
dat.val.X = sparse.model.matrix(~., reducedVal[,-ncol(reducedVal)])[,-1]
dat.finval.X = sparse.model.matrix(~., reducedFinVal[,-ncol(reducedFinVal)])[,-1]



xgb <- xgboost(data = dat.train.X, 
               label = dat.train.y,
               verbose = 1,
               eta = 0.07,
               max_depth = 9, 
               nround=200, 
               subsample = 0.5,
               alpha = 0.03,
               colsample_bytree = 0.5,
               seed = 1,
               objective = "binary:logistic",
               eval_metric = "error",
               scale_pos_weight = sum(dat.train$morethan60kyr == FALSE)/sum(dat.train$morethan60kyr == TRUE)
               )
yhat.xg = predict(xgb , dat.val.X)
yhat.xg = yhat.xg>0.5
sum(yhat.xg == dat.val$morethan60kyr) / nrow(dat.val)


yhat.xg = predict(xgb , dat.finval.X)
yhat.xg = yhat.xg>0.5
sum(yhat.xg == dat.finval$morethan60kyr) / nrow(dat.finval)



names <- dimnames(data.matrix(dat.train.X))[[2]]
importance_matrix <- xgb.importance(names, model = xgb)
xgb.plot.importance(importance_matrix[1:10,])




xgb <- xgb.cv(data = dat.train.X, 
               label = dat.train.y,
               params = 
               eta = 0.07,
               max_depth = 9, 
               nround=200, 
               subsample = 0.5,
               alpha = 0.03,
               colsample_bytree = 0.5,
               seed = 1,
               objective = "binary:logistic",
               eval_metric = "error",
               scale_pos_weight = sum(dat.train$morethan60kyr == FALSE)/sum(dat.train$morethan60kyr == TRUE)
)


yhat = rep(0 , nrow(dat.val))
for (i in 1:nrow(dat.val)) {
  yhat[i] = getmode ( c(yhat.reg.log[i] , yhatForest[i], yhatForest[i], as.logical(yhat.boost$class[i]) , yhatNB[i]  )   )
}

sum(as.logical(yhat)==dat.val$morethan60kyr) / nrow(dat.val)




dat$cma = sapply(dat$cma, as.character)

dat$cma[dat$cma=='Quebec'|dat$cma=='Calgary'|dat$cma=='Toronto'|dat$cma=='Edmonton'|dat$cma=='Hamilton'|dat$cma=='Montreal'|dat$cma=='Ottawa_Gatineau'|dat$cma=='Vancouver'|dat$cma=='Winnipeg'] = 'major' 
dat$cma[dat$cma !='major']  = 'minor'
dat$cma = as.factor(dat$cma)

drops <- c("mfs","nocs")
dat = dat[ , !(names(dat) %in% drops)]




yhatNaive = as.logical(rep(0,nrow(dat.val)))
for(i in (1:nrow(dat.val))){
  prob = 1
  for (j in (1:(ncol(dat.val)-1-length(indexOfNumeric) ))){
    prob = prob * responseRate[[j]][which(levels(dat[,j])==dat.val[i,j])]
  }
  yhatNaive[i] = prob>0.5
}


library(randomForest)
library(mlbench)
library(caret)

customRF <- list(type = "Classification", library = "randomForest", loop = NULL)
customRF$parameters <- data.frame(parameter = c("mtry", "ntree","nodesize"), class = rep("numeric", 3), label = c("mtry", "ntree","nodesize"))
customRF$grid <- function(x, y, len = NULL, search = "grid") {}
customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
  randomForest(x, y, mtry = param$mtry, ntree=param$ntree, nodesize = param$nodesize, ...)
}
customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata)
customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata, type = "prob")
customRF$sort <- function(x) x[order(x[,1]),]
customRF$levels <- function(x) x$classes


# train model
seed = 132
control <- trainControl(method="repeatedcv", number=6, repeats=1)
#tunegrid <- expand.grid(.mtry=c(3:7), .ntree=c(201,301,501,601,801,1001) , .nodesize = seq(8,25,2) )
tunegrid <- expand.grid(.mtry=c(4,5), .ntree=c(30,50) , .nodesize = c(2,10) )
set.seed(seed)
metric <- "Accuracy"
custom <- train(morethan60kyr~., data=dat.train, method=customRF, metric=metric, tuneGrid=tunegrid, trControl=control)
summary(custom)
plot(custom)









logreg = glm(morethan60kyr~. , data = dat.train , family = 'binomial')

yhat.logistic = predict(logreg , newdata = dat.val)
yhat.logistic [yhat.logistic>0.5] = 1
yhat.logistic[yhat.logistic<=0.5] = 0

sum(yhat.logistic == dat.val$morethan60kyr)/length(dat.val$morethan60kyr)


#logistic regression with regularizer
dat.train.X  = model.matrix(~., dat.train[,-d])
dat.train.y = as.factor((dat.train[,d]))
dat.val.X = model.matrix(~., dat.val[,-d])
library(glmnet)
reg.log = cv.glmnet(x=dat.train.X , y=dat.train.y, family = 'binomial' , nfolds = 10 , alpha = 1 )
yhat.reg.log = predict(reg.log , newx = dat.val.X , type = 'response' , s=reg.log$lambda.1se , prob = TRUE)
yhat.reg.log = yhat.reg.log > 0.5
sum(yhat.reg.log == dat.val$morethan60kyr) / length(dat.val$morethan60kyr)

lowCut = 0.17
highCut = 0.83
yhat.ens = rep(0,nrow(dat.val))
yhat.ens[(yhat.reg.log <= highCut) & (yhat.reg.log >= lowCut) ] = yhatForest[(yhat.reg.log <= highCut) & (yhat.reg.log >= lowCut) ]
yhat.ens[(yhat.reg.log > highCut)]  = 1
yhat.ens[((yhat.reg.log < lowCut))] = 0
sum(yhat.ens == dat.val$morethan60kyr) / length(dat.val$morethan60kyr)


summary(yhat.reg.log[which(yhatForest != dat.val$morethan60kyr)])


library(rpart)
treeobj = rpart(morethan60kyr~. , data = dat.train , method = 'class')
yhat.val =  predict(treeobj , newdata = dat.val)
indecies = yhat.val[,2] > yhat.val[,1]
yhat.val2 = rep(0 , nrow(dat.val))
yhat.val2[indecies ] = 1
head(yhat.val2)
sum(yhat.val2 == dat.val$morethan60kyr)/length(dat.val$morethan60kyr)



dat.train$morethan60kyr = as.factor(dat.train$morethan60kyr)
parameterToTune = 1:21
accuracy = rep(0, length(parameterToTune))
for (i in 1: length(parameterToTune)){
  library(randomForest)
  forest  = randomForest(morethan60kyr~. , data = dat.train ,
                         method = 'class', ntree = 801 , nodesize=20)
  yhatForest = predict(forest, newdata = dat.val)
  yhatForest = as.logical(yhatForest)
  sum(yhatForest == dat.val$morethan60kyr)/length(dat.val$morethan60kyr)
  #accuracy[i] = sum(yhatForest == dat.val$morethan60kyr)/length(dat.val$morethan60kyr)
  print(accuracy[i])
}
plot(parameterToTune , accuracy)


yhat = rep(0 , length(dat.val$morethan60kyr))
yhat[yhat.logistic>0.9] = 1
yhat[ (yhat.logistic<=0.9) & (yhat.logistic>0.1)] = yhatForest[(yhat.logistic<=0.9) & (yhat.logistic>0.1)]

sum(yhat == dat.val$morethan60kyr)/length(dat.val$morethan60kyr)



library(adabag)
dat.train$morethan60kyr = as.factor(dat.train$morethan60kyr)
boost.model <- boosting(morethan60kyr ~ ., data=dat.train, boos=FALSE, mfinal=500,
                    coeflearn = 'Freund',
                    control=rpart.control(maxdepth=5))

yhat.boost = predict(boost.model, newdata = dat.val , method = 'class')
sum(yhat.boost$class == dat.val$morethan60kyr)/length(dat.val$morethan60kyr)


indeciesForestGotWrong = which(yhatForest != dat.val$morethan60kyr)
indeciesBoostGotWrong = which(yhat.boost$class != dat.val$morethan60kyr)
indeciesKnnGotWrong = which(knnhat != dat.val$morethan60kyr )
indeciesLogRegGotWrong = which(yhat.reg.log != dat.val$morethan60kyr)







library(class)
p = dim(dat.train)[2];
knnTrain = model.matrix(~. , dat.train[,-p])[,-1];
knnTrain = knnTrain[,indexOfImportant]
knnVal = model.matrix(~. , dat.val[,-p])[,-1];
knnVal = knnVal[,indexOfImportant]
parameterToTune = c(20)
accuracy = rep(0,length(parameterToTune))
for ( i in 1:length(parameterToTune)){
  knnhat = knn(train=knnTrain, test=knnVal, cl=dat.train$morethan60kyr, k = parameterToTune[i])
  accuracy[i] = sum(knnhat == dat.val$morethan60kyr)/length(dat.val$morethan60kyr)
  print(accuracy[i])
}





fsfds







logreg = glm(morethan60kyr~agegrp+hrswrk+nochrd+naics+sex+locstud+mfs , data = dat.train , family = 'binomial')

yhat.logistic = predict(logreg , newdata = dat.val)
yhat.logistic [yhat.logistic>0.5] = 1
yhat.logistic[yhat.logistic<=0.5] = 0

sum(yhat.logistic == dat.val$morethan60kyr)/length(dat.val$morethan60kyr)









cutoff = 0.9
yhat = rep(0 , nrow(dat.val))
predMat = matrix(c(as.logical(knnhat), as.logical(yhatForest) , as.logical(yhat.boost$class)) , nrow = length(yhatForest)  );
for (i in 1:length(dat.val$morethan60kyr)){
  yhat[i] = getmode(predMat[i,])
}

#yhat[attr(knnhat,'prob') > cutoff] = as.logical(knnhat)[attr(knnhat,'prob') > cutoff]
#yhat[attr(knnhat,'prob') <= cutoff] = as.logical(yhatForest[attr(knnhat,'prob') <= cutoff])
sum(as.numeric(yhat == dat.val$morethan60kyr))/length(dat.val$morethan60kyr)




exDat = dat$morethan60kyr [dat$sex=='female' & dat$hhsize>4  ]
exDat = dat$morethan60kyr [dat$attsch == 'Attended'  ]
summary(exDat)
dwed



for(i in unique(dat$naics)){
  print (i)
  print(sum(dat$morethan60kyr[dat$naics == i] )/sum(dat$naics == i))
  }





freqs = calculateFrequencies(dat[,c(indexOfCategorical)])
respnseRate =calculateResponseRate (dat[,c(indexOfCategorical , indexOfResponse)])
par(mfrow= c(1,1) , mar = c(2,8,4,1) + 0.1)
for (i in 1:length(freqs)){
  srted = sort(freqs[[i]] , index = TRUE)
  barplot(srted$x , 
          names.arg = levels(dat[,indexOfCategorical[i]])[srted$ix],las=1 , 
          horiz = TRUE , cex.names = 0.45 , main = names(dat[,indexOfCategorical])[i])
  srted = sort(respnseRate[[i]] , index = TRUE)
  barplot(srted$x , 
          names.arg = levels(dat[,indexOfCategorical[i]])[srted$ix],las=1 , 
          horiz = TRUE , cex.names = 0.45 , main = names(dat[,indexOfCategorical])[i])
}



dat.test = read.csv('datasense_test.csv')
dat.test = dat.test[,-1]
test.freq = calculateFrequencies(dat.test[,c(indexOfCategorical)] )
for (i in 1:length(freqs)){
  mattt = rbind(test.freq[[i]] , freqs[[i]])
  barplot(mattt, 
          names.arg = levels(dat[,indexOfCategorical[i]]),las=1 , 
          horiz = TRUE , cex.names = 0.45 , main = names(dat[,indexOfCategorical])[i]
          ,beside = TRUE)
  
}




sum(dat$morethan60kyr[dat$hdgree == '13'  & dat$sex == 'female'] )/sum(dat$hdgree == '13' & dat$sex == 'female')

sum(dat$morethan60kyr[dat$naics == 'Mining_and_oil_and_gas_extraction' ] )/sum(dat$naics == 'Mining_and_oil_and_gas_extraction')
#[1] 0.7844311




ind =( (dat.train$naics == "Mining_and_oil_and_gas_extraction")&(dat.train$pr == "BritishColumbia") )
sum(as.logical(dat.train$morethan60kyr[ind]) )/ sum(ind)




ind =(  (dat.train$nocs == "Teachers_and_professors" ) & (dat.train$hdgree == 13) )
sum(as.logical(dat.train$morethan60kyr[ind]) )/ sum(ind)


ind =(  (dat.train$naics == "Public_administration") & (dat.train$nochrd == "Supervisor") )
sum(ind)
sum(as.logical(dat.train$morethan60kyr[ind]) )/ sum(ind)


ind =(  (dat.train$hhsize > 1) &(dat.train$marst=="Single")  )
sum(ind)
sum(as.logical(dat.train$morethan60kyr[ind]) )/ sum(ind)


ind =(  (dat$hdgree == 13)  )
sum(ind)
sum(as.logical(dat$morethan60kyr[ind]) )/ sum(ind)


for (i in 1:ncol(dat) ){
  print(cat(i, "-", names(dat)[i] , " " , class(dat[,i]) , length(levels(dat[,i])) ))
}