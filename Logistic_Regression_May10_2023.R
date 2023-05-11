# Script Name:      Logistic_Regression_April_24_2023
# Created On:       April_24_2023
# Author:           Martin_Calvino
# Purpose:          Use logistic regression to predict the outcome of home loan applications (accepted or denied)
# Version:          V3_May10_2023 >> this a follow up study of my previous work:
#                   https://martincalvino.wixsite.com/calvino-art/post/logistic-regression-contingency-tables-tests-of-independence-summary-statistics-in-r   

                    
# OBJECTIVE
# Based on income, loan amount, down payment and ethnicity of home loan applicants >> predict the outcome of loans (accepted or denied)
# for Bank of America nationwide

# Dataset was obtained from the Home Mortgage Disclosure Act (HMDA) - Consumer Financial Protection Bureau (CFPB)
# https://ffiec.cfpb.gov/data-browser/data/2018?category=states&leis=B4TYDEB6GKMZO031MB27

# Documentation of data fields can be found at:
# https://ffiec.cfpb.gov/documentation/2018/lar-data-fields/

# load libraries
library(tidyverse)
library(psych)
library(gplots)
library(vcd)
library(graphics)
library(yardstick)

# load datasets
# Bank of America as 'boa' for years 2018, 2019, 2020 and 2021 respectively
boa18 <- read.csv(file.choose())
boa19 <- read.csv(file.choose())
boa20 <- read.csv(file.choose())
boa21 <- read.csv(file.choose()) # 368,728 observations x 99 variables

# create one big data frame with data from the four years
train.boa <- rbind(boa18, boa19, boa20, boa21) # 1,671,302 observations x 99 variables

colnames(train.boa)

# choose home loan applications with loan_to_value ratios greater than 0% and less than 100%
train.boa <- filter(train.boa, loan_to_value_ratio <= 100)

# feature engineering: estimate down payment as 100 - loan_to_value_ratio
# down payment is in percentual points
train.boa <- mutate(train.boa, downpayment = round(100 - loan_to_value_ratio))
head(train.boa[, 100],  n = 25) # dow npayment in now variable number 100

# feature selection: consider action_taken, income, down payment, loan amount, ethnicity and state variables only
# action_taken is a variable denoting the action of the bank on the home loan application (example: originated or denied)
actionTaken <- train.boa$action_taken
income <- train.boa$income
downpayment <- train.boa$downpayment
loanAmount <- train.boa$loan_amount
ethnicity <- train.boa$derived_ethnicity
state <- train.boa$state_code

# income and downpayment and ethnicity as 'idoet'
train.boa.idoet <- data.frame(income, downpayment, loanAmount, ethnicity, actionTaken, state)
head(train.boa.idoet)

# recode ethnicity variable
train.boa.idoet$ethnicity[train.boa.idoet$ethnicity == "Ethnicity Not Available"] <- NA
head(train.boa.idoet, n = 25)

# how many missing values?
sum(is.na(train.boa.idoet)) # 220,515 NAs
# remove NAs
train.boa.idoet <- na.omit(train.boa.idoet)

# inspect summary statistics
summary(train.boa.idoet[, 1:3])

# multiply income*1000 and choose loan applications with income > 0
train.boa.idoet$income <- train.boa.idoet$income*1000
train.boa.idoet <- filter(train.boa.idoet, income > 0)

# identify and remove outliers
# income
outliers.inc.train.boa.idoet <- boxplot(train.boa.idoet[, 1])$out
train.boa.idoet <- train.boa.idoet[-which(train.boa.idoet[, 1] %in% outliers.inc.train.boa.idoet), ]
boxplot(train.boa.idoet[, c(1:3)])$out
# loan amount
outliers.la.train.boa.idoet <- boxplot(train.boa.idoet[, 3])$out
train.boa.idoet <- train.boa.idoet[-which(train.boa.idoet[, 3] %in% outliers.la.train.boa.idoet), ]
boxplot(train.boa.idoet[, c(1:3)])$out

# inspect summary statistics now that outliers were removed
summary(train.boa.idoet[, 1:3])

# look at ethnicity variable
levels(factor(train.boa.idoet$ethnicity))

# select Hispanic and Not Hispanic home loan applicants
selected_ethnicities <- c("Hispanic or Latino", "Not Hispanic or Latino")
train.boa.idoet <- filter(train.boa.idoet, ethnicity %in% selected_ethnicities)
levels(factor(train.boa.idoet$ethnicity))

# look at the actionTaken variable
levels(factor(train.boa.idoet$actionTaken))
# select loans that were accepted (actionTaken == 1) or denied (actionTaken == 3) 
train.boa.idoet <- filter(train.boa.idoet, actionTaken == 1 | actionTaken == 3)
# recode actionTaken variable
train.boa.idoet$actionTaken[train.boa.idoet$actionTaken == 1] <- "accepted"
train.boa.idoet$actionTaken[train.boa.idoet$actionTaken == 3] <- "denied"
head(train.boa.idoet, n = 25)

str(train.boa.idoet) # 914,315 observations x 6 variables

# explore the relationship among variables
exrelva <- gather(train.boa.idoet, key = "Variable", value = "Value", -actionTaken)
View(exrelva)

# violin plots
exrelva %>%
  filter(Variable != "ethnicity" & Variable != "state") %>%
  ggplot(aes(actionTaken, as.numeric(Value))) +
  facet_wrap(~ Variable, scales = "free_y") +
  geom_violin(draw_quantiles = c(0.25, 0.5, 0.75)) +
  theme_bw()

# bar plot > in Puerto Rico all home loan applications were denied
exrelva %>%
  filter(Variable == "state") %>%
  ggplot(aes(Value, fill = actionTaken)) +
  geom_bar(position = "fill") +
  theme_bw()

exrelva %>%
  filter(Variable == "ethnicity") %>%
  ggplot(aes(Value, fill = actionTaken)) +
  geom_bar(position = "fill") +
  theme_bw()

# remove home loan applications from Puerto Rico because they don't contribute with
# predictive value in regards to the ethnicity of the applicant and the loan being accepted or denied
train.boa.idoet$state[train.boa.idoet$state == "PR"] <- NA
train.boa.idoet <- na.omit(train.boa.idoet)

# code response variable (actionTaken) as 0 and 1 to implement logistic regression
train.boa.idoet$actionTaken[train.boa.idoet$actionTaken == "accepted"] <- 1
train.boa.idoet$actionTaken[train.boa.idoet$actionTaken == "denied"] <- 0
train.boa.idoet$actionTaken <- factor(
  train.boa.idoet$actionTaken,
  levels = c(0, 1),
  labels = c(0, 1)
)

table(train.boa.idoet$actionTaken) # 408,547 denied loans and 505,767 accepted loans

# use a train/test split
rows <- sample(nrow(train.boa.idoet))
train.boa.idoet <- train.boa.idoet[rows, ]
# train:test split is 70%:30%
split <- round(nrow(train.boa.idoet) * 0.70)
train <- train.boa.idoet[1:split, ]
test <- train.boa.idoet[(split + 1):nrow(train.boa.idoet), ]
nrow(train) / nrow(train.boa.idoet) # training dataset is 70% of the entire dataset


# fit a logistic regression model to training data using the glm() function
fit.boa <- glm(actionTaken ~ income + downpayment + loanAmount + ethnicity + state,
               data = train, family = binomial())

# inspect model's coefficients
summary(fit.boa)

# make predictions on the test dataset to evaluate model performance
pred.test <- predict(fit.boa, newdata = test, type = "response")
summary(pred.test)

# evaluate model performance
# Confusion Matrix: counts of outcomes
actual_response <- test$actionTaken
predicted_response <- ifelse(pred.test > 0.50, "1", "0")
outcomes <- table(predicted_response, actual_response)
outcomes

# evaluate model using functions from the yardstick package
confusion <- conf_mat(outcomes)
autoplot(confusion)
# model performance metrics
summary(confusion,event_level = "second")

# accuracy is the proportion of correct predictions > 0.685
# sensitivity is the proportion of true positives > 0.702
# specificity is the proportion of true negatives > 0.665


# use the model to make predictions on new data
# what is the probability that Bank of America will accept a home loan applications from
# a Latino earning $55,000/year with a down payment of 10%, borrowing $517,500 and living in Arizona?
predict(fit.boa, newdata = data.frame(income = 55000, downpayment = 10, loanAmount = 517500,
                                      ethnicity = "Hispanic or Latino", state = "AR"),
        type = "response")


