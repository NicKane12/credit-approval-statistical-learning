# =====================================================
# Predicting Credit Approval Using Statistical Learning
# Author: Nic Kane
# =====================================================

# -------------------------
# Libraries
# -------------------------
library(dplyr)
library(tidyr)
library(ROSE)
library(ggplot2)
library(boot)
library(MASS)
library(Epi)
library(pROC)
library(lattice)
library(DAAG)
library(class)
library(tree)
library(randomForest)
library(glmnet)
library(gbm)


# -------------------------
# Data Loading & Inspection
# -------------------------
credit_data <- read.csv("CreditCard.csv")
head(credit_data)
glimpse(credit_data)

# -------------------------
# Data Cleaning & Preprocessing
# -------------------------
credit_clean = credit_data[,-1]
summary(credit_clean)


colSums(is.na(credit_clean))

#Sorting
sort(credit_data$age)

credit_clean <- subset(credit_clean, age >= 18)

#Rounding some values
credit_clean$income <- round(credit_clean$income, digits = 2)
credit_clean$expenditure <- round(credit_clean$expenditure, digits = 2)
credit_clean$age <- round(credit_clean$age + .01)
credit_clean$share <- round(credit_clean$share, digits = 6)


#changing the categorical variables to 1 or 0
credit_clean <- credit_clean %>%
  mutate(
    card = ifelse(card == "yes", 1, 0),
    owner = ifelse(owner == "yes", 1, 0),
    selfemp = ifelse(selfemp == "yes", 1, 0)
  )

credit_clean$card <-as.factor(credit_clean$card)

summary(credit_clean)

# -------------------------
# Class Imbalance Handling
# -------------------------
table(credit_clean$card)
prop.table(table(credit_clean$card))

set.seed(1)
credit_clean_under = ovun.sample(card ~ ., data=credit_clean, method="under")$data

table(credit_clean_under$card)
prop.table(table(credit_clean_under$card))

pairs(credit_clean_under)

# -------------------------
# Exploratory Data Analysis
# -------------------------
ggplot(credit_clean_under, aes(x = card, y = reports, fill = card)) +
  geom_boxplot() +
  labs(title = "Reports by Card Ownership")

ggplot(credit_clean_under, aes(x = card, y = age, fill = card)) +
  geom_boxplot() +
  labs(title = "Age by Card Ownership")

ggplot(credit_clean_under, aes(x = card, y = income, fill = card)) +
  geom_boxplot() +
  labs(title = "Income by Card Ownership")

#See relationship between share and card
ggplot(credit_clean_under, aes(x = card, y = share, fill = card)) +
  geom_boxplot() +
  labs(title = "Share by Card Ownership")

max(credit_clean_under$share[credit_clean_under$card == 0])

ggplot(credit_clean_under, aes(x = card, y = dependents, fill = card)) +
  geom_boxplot() +
  labs(title = "Dependents by Card Ownership")

ggplot(credit_clean_under, aes(x = card, y = months, fill = card)) +
  geom_boxplot() +
  labs(title = "Months by Card Ownership")

ggplot(credit_clean_under, aes(x = card, y = majorcards, fill = card)) +
  geom_boxplot() +
  labs(title = "MajorCard by Card Ownership")

ggplot(credit_clean_under, aes(x = card, y = active, fill = card)) +
  geom_boxplot() +
  labs(title = "Active by Card Ownership")

ggplot(credit_clean_under, aes(x = card, y = expenditure, fill = card)) +
  geom_boxplot() +
  labs(title = "Expenditure by Card Ownership")

any(credit_clean_under$card == 0 & credit_clean_under$expenditure != 0)
credit_clean_under = credit_clean_under[,-6]
# Remove expenditure to prevent leakage (not available at application time) 

ggplot(credit_clean_under, aes(x = owner, fill = card)) +
  geom_bar(position = "fill") +
  labs(title = "Card status by Home Ownership",
       x = "Owner", y = "Proportion")

ggplot(credit_clean_under, aes(x = selfemp, fill = card)) +
  geom_bar(position = "fill") +
  labs(title = "Card status by Self Employed",
       x = "Owner", y = "Proportion")

#strong
#reports, active, income, owner

#weak
# selfemp, months, age

#leakage
#expenditure, share, reports

# -------------------------
# Train / Test Split
# -------------------------
credit_modeling = credit_clean_under[, c(1,2,4,6,11)]

set.seed(12)
train_index <- sample(seq_len(nrow(credit_modeling)), size = 0.8 * nrow(credit_modeling))
train <- credit_modeling[train_index,]
test <- credit_modeling[-train_index,]

# -------------------------
# Modeling & Evaluation
# -------------------------

# Logistic Regression
log.fit1 <- glm(card ~ reports + income + owner + active,
                data = train, family = binomial
  
)

summary(log.fit1)

set.seed(1)
stepAIC(log.fit1, direction="backward")
stepAIC(log.fit1, direction="forward")

log.fit2 <- glm(card ~ reports + income + active,
                data = train, family = binomial)

log.prob <- predict(log.fit2, test, type="response")
log.pred <- rep(0, length(test$card))
log.pred[log.prob > 0.3] = 1
table(log.pred, test$card)

log.accuracy <- mean(log.pred == test$card)
log.accuracy

log_dat <- data.frame(
  card = test$card, score = log.prob)

ROC(form = card ~ score, data = log_dat, plot = "ROC")

out0=cv.glm(data = credit_modeling, glmfit = log.fit2) 
cost<-function(r,pi=0) {
  mean(abs(r-pi)>0.5)}
out1=cv.glm(data = credit_modeling, glmfit = log.fit2, cost, K=10)
cv.binary(log.fit2)
1-out0$delta[2] #loocv
1-out1$delta[2] #10-fold


# LDA
lda.fit <- lda(card ~ reports + income + owner + active, data = train)
lda.pred <- predict(lda.fit, newdata = test)
confusion <- table(Predicted = lda.pred$class, Actual = test$card)
confusion
lda.accuracy <- mean(lda.pred$class == test$card)
lda.accuracy


prob.l <- lda.pred$posterior[, 2] 
lda_dat <- data.frame(card = test$card, score = prob.l)
lda_roc <- ROC(form = card ~ score,
               data = lda_dat,
               plot = "ROC")

# QDA
qda.fit <- qda(card ~ reports + income + owner + active, data = train)
qda.pred <- predict(qda.fit, newdata = test)
confusion <- table(Predicted = qda.pred$class, Actual = test$card)
confusion
qda.accuracy <- mean(qda.pred$class == test$card)
qda.accuracy

prob.q <- qda.pred$posterior[, 2] 

qda_dat <- data.frame(card = test$card, score = prob.q)

lda_roc <- ROC(form = card ~ score, data = qda_dat, plot = "ROC")

# KNN
attach(credit_modeling)
vars <- c("reports", "income", "owner", "active")

train.means <- sapply(train[, vars], mean)
train.sds   <- sapply(train[, vars], sd)

train.X <- scale(train[, vars], center = train.means, scale = train.sds)
test.X  <- scale(test[, vars],  center = train.means, scale = train.sds)

train.card <- train$card
test.card  <- test$card
set.seed(1)

knn.pred=knn(train.X,test.X,train.card,k=1, prob = TRUE) #K=1
table(knn.pred,test.card)
knn1.accuracy <- mean(knn.pred == test.card)
knn1.accuracy

knn.pred=knn(train.X,test.X,train.card,k=3, prob = TRUE) #K=3
table(knn.pred,test.card)
knn3.accuracy  <- mean(knn.pred == test.card)
knn3.accuracy

knn.pred=knn(train.X,test.X,train.card,k=5, prob = TRUE) #K=5
table(knn.pred,test.card)
knn5.accuracy  <- mean(knn.pred == test.card)
knn5.accuracy

prob.raw <- attr(knn.pred, "prob")
pred.num <- as.numeric(knn.pred) - 1
prob.knn <- ifelse(pred.num == 1, prob.raw, 1 - prob.raw)

knn.dat <- data.frame(card = test.card, score = prob.knn)

ROC(form = card ~ score, data = knn.dat, plot = "ROC")

#K=3 is best

# Classification Tree
tree.fit <- tree(card ~ reports + income + owner + active, data = train)
plot(tree.fit)
text(tree.fit, pretty = 0)

tree.pred <- predict(tree.fit, newdata = test, type = "class")
confusion <- table(Predicted = tree.pred, Actual = test$card)
confusion

tree.accuracy <- mean(tree.pred == test$card)
tree.accuracy

tree.prob <- predict(tree.fit, newdata = test, type = "vector")[,2]

tree_dat <- data.frame(card = test$card, score = tree.prob)

ROC(form = card ~ score, data = tree_dat, plot="ROC")

cv.tree.fit <- cv.tree(tree.fit, FUN = prune.misclass)
plot(cv.tree.fit)
best.size <- cv.tree.fit$size[which.min(cv.tree.fit$dev)]

pruned.tree <- prune.misclass(tree.fit, best = best.size)

plot(pruned.tree)
text(pruned.tree, pretty = 0)

# Random Forest
train$card <- as.factor(train$card)
test$card  <- as.factor(test$card)
rf.fit <- randomForest(card ~ reports + income + owner + active, data = train, 
                       ntree = 500, mtry = 2, importance = TRUE)

rf.fit

rf.pred <- predict(rf.fit, newdata = test, type = "class")
table(Predicted = rf.pred, Actual = test$card)

rf.accuracy <- mean(rf.pred == test$card)
rf.accuracy

rf.prob <- predict(rf.fit, newdata = test, type = "prob")[,2]
rf_dat <- data.frame(card = test$card, score = rf.prob)

ROC(form = card ~ score, data = rf_dat, plot = "ROC")

varImpPlot(rf.fit)

# Boosting
credit_modeling$card <- as.numeric(as.character(credit_modeling$card))
train$card <- as.numeric(as.character(train$card))
test$card <- as.numeric(as.character(test$card))

boost.fit <- gbm(formula = card ~ reports + income + owner + active, data = train, distribution = "bernoulli",n.trees = 5000,
  interaction.depth = 3, shrinkage = 0.01, bag.fraction = 0.5, n.minobsinnode = 10, verbose = FALSE)

boost.prob <- predict(boost.fit, newdata = test, n.trees = 5000, type = "response")

boost.pred <- ifelse(boost.prob > 0.5, 1, 0)

confusion <- table(Predicted = boost.pred, Actual = test$card)
confusion

boost.accuracy <- mean(boost.pred == test$card)
boost.accuracy

boost_dat <- data.frame(card = test$card, score = boost.prob)

ROC(form = card ~ score, data = boost_dat, plot = "ROC")


# Ridge/Lasso
X.train <- model.matrix(card ~ reports + income + owner + active, data = train)[ , -1]
X.test  <- model.matrix(card ~ reports + income + owner + active, data = test)[ , -1]

y.train <- train$card
y.test  <- test$card

ridge.cv <- cv.glmnet(X.train, y.train, family = "binomial", alpha = 0)
plot(ridge.cv)

lambda.ridge <- ridge.cv$lambda.min
lambda.ridge
ridge.fit <- glmnet(X.train, y.train, family = "binomial", alpha = 0, lambda = lambda.ridge)
coef(ridge.fit)

ridge.prob <- predict(ridge.fit, newx = X.test, type = "response")
ridge.pred <- ifelse(ridge.prob > 0.5, 1, 0)

confusion <- table(Predicted = ridge.pred, Actual = y.test)
confusion
ridge.accuracy <- mean(ridge.pred == y.test)
ridge.accuracy

ridge_dat <- data.frame(card = y.test, score = ridge.prob)
ROC(form = card ~ s0, data = ridge_dat, plot = "ROC")


lasso.cv <- cv.glmnet(X.train, y.train, family = "binomial", alpha = 1)
lambda.lasso <- lasso.cv$lambda.min

lasso.fit <- glmnet(X.train, y.train, family = "binomial", alpha = 1, lambda = lambda.lasso)
coef(lasso.fit)

lasso.prob <- predict(lasso.fit, newx = X.test, type = "response")
lasso.pred <- ifelse(lasso.prob > 0.5, 1, 0)

confusion <- table(Predicted = lasso.pred, Actual = y.test)
confusion
lasso.accuracy <- mean(lasso.pred == y.test)
lasso.accuracy

lasso.dat <- data.frame(card = y.test, score = lasso.prob)
ROC(form = card ~ s0, data = lasso.dat, plot = "ROC")


# -------------------------
# Model Comparison
# -------------------------
results <- data.frame(
  Model = c("Logistic", "LDA", "QDA", "KNN (3)","Classification Tree", 
            "Random Forests","Boosting", "Ridge", "LASSO"),
  Accuracy = c(0.328, 0.767, 0.672, 0.802, 0.681, 0.801, 0.715, 0.759, 0.759),
  AUC      = c(0.843, 0.802, 0.823, 0.835, 0.812, 0.842, 0.835, 0.828, 0.843))

results_long <- results |>
  pivot_longer(cols = c(Accuracy, AUC),
               names_to = "Metric",
               values_to = "Value")

ggplot(results_long, aes(x = Model, y = Value, fill = Metric)) +
  geom_col(position = "dodge") +
  coord_cartesian(ylim = c(0.3, 0.9)) +
  labs(title = "Model Comparison: Accuracy vs AUC",
       x = "Model", y = "Value") +
  theme_bw() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

sessionInfo()