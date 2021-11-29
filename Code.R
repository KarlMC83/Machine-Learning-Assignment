## Machine Learning Assignment
## Author: Karl Melgarejo Castillo
## Date: 24/11/2021


# 1. Reading data files
theURL1 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
training <- read.csv(theURL1, sep = ",")

theURL2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
testing <- read.csv(theURL2, sep = ",")

"Source: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har"

# 2. Descriptive analysis

dim(training)
str(training)
head(training)

dim(testing)
str(testing)
head(testing)

str(training$classe)
        #there is no "classe" variable in the testing data set
table(training$classe)
        #Five classes: Class A: exactly according to the specification; Class B: throwing the elbows to the front;
                #Class C: lifting the dumbbell only halfway; Class D: lowering the dumbbell only halfway;
                #Class E: throwing the hips to the front.

table(training$user_name)

table(testing$user_name)
names(training)

summary(training$raw_timestamp_part_2)
table(training$cvtd_timestamp)


# Libraries
library(caret)


# Set seed
set.seed(1111)

g <- grep("gyros|accel|magnet", names(training))
imu <- names(training)[g]
g <- grep("var", names(training))
o <- names(training)[g]
imu <- imu[!(imu %in% o)] 
imu

# Lumbar belt
g <- grep("belt", names(training))
length(g)
names(training)[g]
        # Roll, Pitch & Yaw
        id <- c(grep("roll", names(training)[g]), grep("pitch", names(training)[g]),
                grep("yaw", names(training)[g]))
        nb <- names(training)[g][id]
        nb
        
# Forearm
g <- grep("forearm", names(training))
length(g)
names(training)[g]
        # Roll, Pitch & Yaw
        id <- c(grep("roll", names(training)[g]), grep("pitch", names(training)[g]),
                grep("yaw", names(training)[g]))
        nf <- names(training)[g][id]
        nf
        
# Arm
g <- grep("arm", names(training))
length(g)
names(training)[g]
        # Roll, Pitch & Yaw
        id <- c(grep("roll", names(training)[g]), grep("pitch", names(training)[g]),
                grep("yaw", names(training)[g]))
        na <- names(training)[g][id]
        na <- na[!(na %in% nf)]
        
# Dumbbell
g <- grep("dumbbell", names(training))
length(g)
names(training)[g]
        # Roll, Pitch & Yaw
        id <- c(grep("roll", names(training)[g]), grep("pitch", names(training)[g]),
                grep("yaw", names(training)[g]))
        nd <- names(training)[g][id]
        nd

        
#t <- subset(training, select = c("classe", nb, nf, na, nd))        
t <- subset(training, select = c("classe", imu))        
str(t)

t$classe <- as.factor(t$classe)
class(t$classe)


inTrain <- createDataPartition(y=t$classe, p=0.7, list=FALSE)
t_train <- t[inTrain,]
t_test <- t[-inTrain,]
dim(t_train)
dim(t_test)

#qplot(roll_belt, pitch_belt, colour=classe, data=t_train)

pc <- preProcess(t_train[,-1], method="pca", thresh = 0.8)
pc
t_train_pc <- predict(pc, t_train[,-1])
dim(t_train_pc)


# Training estimation
system.time(
        tr1 <- train(y = t_train$classe, x = t_train_pc, method = "rf",
                     trControl = trainControl(method = "cv", number = 10),
                     verbose=FALSE
                        )
        )

system.time(
        tr2 <- train(y = t_train$classe, x = t_train_pc, method = "gbm",
                     trControl = trainControl(method = "cv", number = 10),
                     verbose=FALSE
                        )
       )

system.time(
        tr21 <- train(classe ~., data = t_train, method = "gbm",
                      trControl = trainControl(method = "cv", number = 10),
                      verbose=FALSE
                        )
        )

system.time(
        tr3 <- train(y = t_train$classe, x = t_train_pc, method = "lda",
                     trControl = trainControl(method = "cv", number = 10),
                     verbose=FALSE
                        )
        )

system.time(
        tr31 <- train(classe ~., data = t_train, method = "lda",
                      trControl = trainControl(method = "cv", number = 10),
                      verbose=FALSE
                        )
        )

tr1
tr2
tr21
tr3
tr31

        # PC
        t_test_pc <- predict(pc,t_test[,-1])

pr1 <- predict(tr1, t_test_pc)
pr2 <- predict(tr2, t_test_pc)
pr21 <- predict(tr21, t_test[,-1])
pr3 <- predict(tr3, t_test_pc)
pr31 <- predict(tr31, t_test[,-1])


confusionMatrix(t_test$classe, pr1)$overall["Accuracy"]
confusionMatrix(t_test$classe, pr2)$overall["Accuracy"]
confusionMatrix(t_test$classe, pr21)$overall["Accuracy"]
confusionMatrix(t_test$classe, pr3)$overall["Accuracy"]
confusionMatrix(t_test$classe, pr31)$overall["Accuracy"]


length(pr3)

pre_comb <- data.frame(pr1, pr21,t_test$classe)

str(pre_comb)

tr4 <- train(t_test.classe ~ ., data = pre_comb, method = "rf")
tr4$results$Accuracy[1]
pr4 <- predict(tr4,pre_comb)

confusionMatrix(t_test$classe, pr4)$overall["Accuracy"]

## Test sample
tt <- subset(testing, select = imu)        
names(tt)

# PC
tt_test_pc <- predict(pc,tt)

tt_pr1 <- predict(tr1, tt_test_pc)
tt_pr2 <- predict(tr2, tt_test_pc)
tt_pr21 <- predict(tr21, tt)
tt_pr3 <- predict(tr3, tt_test_pc)
tt_pr31 <- predict(tr31, tt)

table(tt_pr1)
table(tt_pr2)
table(tt_pr21)
table(tt_pr3)
table(tt_pr31)
