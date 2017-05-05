#clearing the memory
rm(list=ls())
#gtetting rid of the plots
dev.off()
plot.new()

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
  d = ncol(dat)-1
  print(d)
  responseInd = d+1
  responseRate = list()
  frequencies = list()
  for (i in 1:d){
    numLevels = length(levels(dat[,i]))
    responseRate[[i]] = rep(0, numLevels)
    frequencies[[i]] = rep(0, numLevels)
    for (j in 1:numLevels ) {
      posInd = dat[,i] == levels(dat[,i])[j]
      prob = sum(dat[,responseInd][posInd ] )/sum(posInd)
      responseRate[[i]][j] = prob 
      frequencies[[i]][j] = sum(posInd)/n
    }
  }
  return (list(frequencies , responseRate))
}


set.seed(143)
dat = read.csv ('datasense_training.csv')
dat = dat[,-1]
n = nrow(dat)
d = ncol(dat)

indexOfCategorical = which(lapply(dat, class) == 'factor')
d.cat = length(indexOfCategorical)
indexOfNumeric = c(which(lapply(dat, class) == 'integer') , which(lapply(dat, class)=='numeric') ) 
indexOfResponse = d



dat$hdgree = as.factor(dat$hdgree)
dat$hrswrk = scale(dat$hrswrk)
dat$agegrp = scale( log(dat$agegrp) )
dat$hhsize = scale(dat$hhsize)

ii = ((1:n) %%10)+1
ii = sample(ii)

dat.train = dat[ii!=1 ,  ]
dat.val = dat[ii==1 , ]



library(mclust)
X = model.matrix (~. , dat.train [ , -d])
dis = dist(X , method = "Euclidean")
hc = hclust(X)


#===============Boosting=================
library(adabag)
dat.train$morethan60kyr = as.factor(dat.train$morethan60kyr)
boost.model <- boosting(morethan60kyr ~ ., data=dat.train, boos=FALSE, mfinal=500,
                        coeflearn = 'Freund',
                        control=rpart.control(maxdepth=5))

yhat.boost = predict(boost.model, newdata = dat.val , method = 'class')
sum(yhat.boost$class == dat.val$morethan60kyr)/length(dat.val$morethan60kyr)






rr = calculateFrequencies(dat.train[,c(indexOfCategorical , indexOfResponse)])
fre = rr[[1]]
responseRate = rr[[2]]

hist(as.numeric(responseRate[[1]]) , breaks = length(responseRate[[1]]))

par(mfrow= c(1,1) , mar = c(2,8,4,1) + 0.1)
for (i in 1:length(responseRate)){
  srted = sort(responseRate[[i]] , index = TRUE)
  barplot(srted$x , 
          names.arg = levels(dat[,indexOfCategorical[i]])[srted$ix],las=1 , 
          horiz = TRUE , cex.names = 0.45 , main = names(dat[,indexOfCategorical])[i])
}

barplot(table( dat.train$morethan60kyr,dat.train$hhsize))


library(e1071)
datNB = dat.train[,-which(names(dat.train) %in% c("mfs" , "nocs" , "pr","fol"))]
NBmodel = naiveBayes( as.factor(morethan60kyr)~. , data= datNB )

yhatNB = predict(NBmodel , newdata = dat.val  )

sum(yhatNB == dat.val$morethan60kyr)/length(dat.val$morethan60kyr)


sum(dat$morethan60kyr[dat.train$hdgree == '13'  & dat.train$sex == 'female'] )/sum(dat.train$hdgree == '13' & dat.train$sex == 'female')











