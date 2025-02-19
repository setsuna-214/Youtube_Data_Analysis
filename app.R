# The Shiny App code is starts at line 515.

library(ggplot2)
library(dplyr)
library(gridExtra)
library(mice)
library(tibble)
library(tidyr)
library(crayon)
library(knitr)
library(lattice)
library(klaR)
library(lime)
library(ROCR)
library(ROCit)
library(rpart)
library(rpart.plot)
library(pander)
library(recipes)
library(e1071)
library(xgboost)
library(corrplot)
library(fpc)
library(caret)

gy_data <- read.csv("processed_Youtube_data.csv")

# Create a new column 'earnings' based on the median
gy_data$earnings <- ifelse(gy_data$lowest_yearly_earnings > 49430.3, 1, 0)

# Count the number of 1 and 0 in 'earnings'
count_earnings_1 <- sum(gy_data$earnings == 1)
count_earnings_0 <-sum(gy_data$earnings == 0)
count_earnings_1
count_earnings_0

myvars <- c( "video.views", "video_views_for_the_last_30_days", "subscribers", "subscribers_for_last_30_days", "Population", "Urban_population", "earnings")
df_to_classification <- gy_data[myvars]

df_to_classification$subscribers_for_last_30_days <- ifelse(df_to_classification$subscribers_for_last_30_days == "other", NA, df_to_classification$subscribers_for_last_30_days)
df_to_classification$subscribers_for_last_30_days <- as.numeric(df_to_classification$subscribers_for_last_30_days)
df_to_classification$video.views <- as.numeric(df_to_classification$video.views)
df_to_classification$video_views_for_the_last_30_days <- as.numeric(df_to_classification$video_views_for_the_last_30_days)

# set seeds
set.seed(12345)
intrain <- runif(nrow(df_to_classification)) < 0.80
train <- df_to_classification[intrain,]
test <- df_to_classification[!intrain,]

# split the train set into training set and validation set
pos <- 1
outcome <- "earnings"
useForVal <- rbinom(n=dim(train)[1], size=1, prob=0.1)>0
dCal <- subset(train, useForVal)
dTrain <- subset(train, !useForVal)
useForVal <- rbinom(n=dim(train)[1], size=1, prob=0.1)>0
validation <- subset(train, useForVal)
train <- subset(train, !useForVal)

PredC <- function(outColumn, varColumn, appColumn) {
  pPos <- sum(outColumn == pos) / length(outColumn)
  naTab <- table(outColumn[is.na(varColumn)])
  pPosWna <- (naTab/sum(naTab))[pos]
  vTab <- table(outColumn, varColumn)
  pPosWv <- (vTab[pos,] + 1.0e-3 * pPos) / (colSums(vTab) + 1.0e-3)
  pred <- pPosWv[appColumn]
  pred[is.na(appColumn)] <- pPosWna
  pred[is.na(pred)] <- pPos
  pred
}

earnsVars <- setdiff(myvars, 'earnings')

for(v in earnsVars) {
  p <- paste('pred_', v, sep = '')
  train[, p] <- PredC(train[,outcome], train[, v], train[,v])
  validation[, p] <- PredC(train[,outcome], train[, v], validation[,v])
  test[, p] <- PredC(train[, outcome], train[,v], test[,v])
}

calcAUC <- function(predCol, outCol) {
  perf <- performance(prediction(predCol, outCol == pos), 'auc')
  as.numeric(perf@y.values)
}

for (v in earnsVars) {
  p <- paste('pred_', v, sep = '')
  aucTrain <- calcAUC(train[, p], train[, outcome])
  if (aucTrain >= 0.5) {
    aucCal <- calcAUC(validation[,p], validation[,outcome])
    print(sprintf(
      "%s: trainAUC: %5.3f; validationAUC: %4.3f",
      p, aucTrain, aucCal))
  }
}

vars <- c("video.views", "video_views_for_the_last_30_days", "subscribers", "subscribers_for_last_30_days", "Population", "Urban_population")

for (var in vars) {
  aucs <- rep(0, 100)
  for (rep in 1:length(aucs)) {
    useForCalRep <- rbinom(n = nrow(train), size = 1, prob = 0.1)>0
    predRep <- PredC(train[!useForCalRep, outcome],
                     train[!useForCalRep, var],
                     train[useForCalRep, var])
    aucs[rep] <- calcAUC(predRep, train[useForCalRep, outcome])
  }
  print(sprintf("%s: mean: %5.3f; sd: %5.3f", var, mean(aucs), sd(aucs)))
}

# Calculate and store AUC values for each single variable
auc_values <- numeric(length = length(earnsVars))
for (i in 1:length(earnsVars)) {
  v <- earnsVars[i]
  p <- paste('pred_', v, sep = '')
  auc_values[i] <- calcAUC(test[, p], test[, outcome])
}

# Sort AUC values and corresponding variables
top_four_indices <- tail(order(auc_values, decreasing = TRUE), 4)
top_four_variables <- earnsVars[top_four_indices]

# Set up a blank plot with appropriate axes and labels
plot(NA, NA, xlim = c(0, 1), ylim = c(0, 1), xlab = "False Positive Rate", ylab = "True Positive Rate", main = "Combined ROC Curves")

# Plot ROC curves for the top four variables
library(ROCR) # Make sure you have ROCR package installed
colors <- c("red", "green", "blue", "purple")
for (i in 1:4) {
  v <- top_four_variables[i]
  p <- paste('pred_', v, sep = '')
  perf <- performance(prediction(test[, p], test[, outcome] == pos), "tpr", "fpr")
  plot(perf, col = colors[i], add = TRUE, lwd = 2, lty = i)
  legend(0, 1, legend = top_four_variables, col = colors, lty = 1:4, title = "Variables", cex = 0.8)
}

Npos <- sum(train[,outcome] == 1)
pred.Null <- Npos / nrow(train)
cat("Propotion of outcome equals to 1 in the training dataset:", pred.Null)

TP <- 0; TN <- sum(validation[, outcome] == 1); 
FP <- 0; FN <- sum(validation[, outcome] == 0); 

accuracy <- (TP + TN) / nrow(validation)
cat(accuracy)

precision <- TP/(TP + FP)
cat(precision)

recall <- TP/(TP + FN)
cat(recall)

pred.Null <- rep(pred.Null, nrow(validation))
AUC <- calcAUC(pred.Null, validation[, outcome])

logLikelihood <- function(ytrue, ypred, epsilon = 1e-6) {
  sum(ifelse(ytrue == pos, log(ypred + epsilon), log(1-ypred-epsilon)), na.rm = T)
}

logNull <- logLikelihood(train[,outcome], sum(train[,outcome]==pos)/nrow(train))
cat(logNull)

selVars <- c()
selPredVars <- c()
minDrop <- 10

for (v in earnsVars) {
  pi <- paste('pred', v, sep = '_') 
  devDrop <- 2 * (logLikelihood(validation[, outcome], validation[, pi]) - logNull)
  if (devDrop >= minDrop) {
    print(sprintf("%s, deviance reduction: %g", pi, devDrop))
    selPredVars <- c(selPredVars, pi)
    selVars <- c(selVars, v)
  }
}

fv <- paste(outcome,' ~ ',paste(earnsVars, collapse="+"),sep='')

tmodel<- rpart(fv,data=train)

logLikelihood <- function(ytrue, ypred, epsilon=1e-6) {
  sum(ifelse(ytrue, log(ypred+epsilon), log(1-ypred+epsilon)), na.rm=T) }

performanceMeasures <- function(ytrue, ypred, model.name = "model", threshold = 0.5) {
  dev.norm <- -2 * logLikelihood(ytrue, ypred)/length(ypred)
  
  cmat <- table(actual = ytrue, predicted = ypred)
  accuracy <- sum(diag(cmat)) / sum(cmat)
  precision <- cmat[2, 2] / sum(cmat[, 2])
  recall <- cmat[2, 2] / sum(cmat[2, ])
  f1 <- 2 * precision * recall / (precision + recall)
  data.frame(model = model.name, precision = precision,
             recall = recall, f1 = f1, dev.norm = dev.norm)
}

panderOpt <- function(){
  panderOptions("plain.ascii", TRUE)
  panderOptions("keep.trailing.zeros", TRUE)
  panderOptions("table.style", "simple")
}

pretty_perf_table <- function(model, xtrain, ytrain, xval, yval, xtest, ytest, threshold=0.5) {
  
  panderOpt()
  perf_justify <- "lrrrr"
  
  # Directly obtain the numeric vector of predictions
  pred_train <- predict(model, newdata=xtrain)
  pred_val <- predict(model, newdata=xval)
  pred_test <- predict(model, newdata=xtest)
  
  # Use the threshold to get the predicted classifications
  trainperf_df <- performanceMeasures(ytrain, pred_train >= threshold, model.name="training")
  valperf_df <-  performanceMeasures(yval, pred_val >= threshold, model.name="validation")
  testperf_df <- performanceMeasures(ytest, pred_test >= threshold, model.name="test")
  
  perftable <- rbind(trainperf_df, valperf_df, testperf_df)
  pandoc.table(perftable, justify = perf_justify)
}

# Now you can call the function as you provided earlier
pretty_perf_table(tmodel, 
                  train[earnsVars], train[,outcome]==pos,
                  validation[earnsVars], validation[,outcome]==pos,
                  test[earnsVars], test[,outcome]==pos)

log_tmodel <- logLikelihood(train[,outcome]==pos, predict(tmodel, newdata=train))

plot_roc <- function(predcol1, outcol1, predcol2, outcol2){
  # Calculate ROC for both training and validation data
  roc_train <- rocit(score=predcol1, class=outcol1==pos)
  roc_val <- rocit(score=predcol2, class=outcol2==pos)
  
  # Plot ROC for training data
  plot(roc_train, col="blue", lwd=3, legend=FALSE, YIndex=FALSE, values=TRUE, asp=1, main="ROC Curve")
  
  # Add ROC for validation data
  lines(roc_val$TPR ~ roc_val$FPR, col="red", lwd=3)
  
  # Add legend to differentiate between the curves
  legend("bottomright", col=c("blue", "red"), legend=c("Training Data", "Validation Data"), lwd=1.5, cex=0.8, inset=c(0.1, 0.1))
}

# Now, use this function to plot the ROC curves
pred_train_roc <- predict(tmodel, newdata=train)
pred_val_roc <- predict(tmodel, newdata=validation)

p1 <- plot_roc(pred_train_roc, train[[outcome]], pred_val_roc, validation[[outcome]])

tVars <- paste('pred_', earnsVars, sep='')

fv2 <- paste(outcome, ' ~ ', paste(tVars, collapse = '+'), sep = '')

tmodel2 <- rpart(fv2, data=train)

pretty_perf_table(tmodel2, 
                  train[tVars], train[,outcome]==pos,
                  validation[tVars], validation[,outcome]==pos,
                  test[tVars], test[,outcome]==pos)

plot_roc2 <- function(predcol1, outcol1, predcol2, outcol2){
  # Calculate ROC for both training and validation data
  roc_train <- rocit(score=predcol1, class=outcol1==pos)
  roc_val <- rocit(score=predcol2, class=outcol2==pos)
  
  # Plot ROC for training data
  plot(roc_train, col="darkgreen", lwd=3, legend=FALSE, YIndex=FALSE, values=TRUE, asp=1, main="ROC Curve")
  
  # Add ROC for validation data
  lines(roc_val$TPR ~ roc_val$FPR, col="orange", lwd=3)
  
  # Add legend to differentiate between the curves
  legend("bottomright", col=c("darkgreen", "orange"), legend=c("Training Data", "Validation Data"), lwd=1.5, cex=0.8, inset=c(0.1, 0.1))
}

# Now, use this function to plot the ROC curves
pred_train_roc <- predict(tmodel2, newdata=train)
pred_val_roc <- predict(tmodel2, newdata=validation)

p2 <- plot_roc2(pred_train_roc, train[[outcome]], pred_val_roc, validation[[outcome]])

median_value <- median(df_to_classification$subscribers_for_last_30_days, na.rm = TRUE)


train$subscribers_for_last_30_days[is.na(train$subscribers_for_last_30_days)] <- median_value
test$subscribers_for_last_30_days[is.na(test$subscribers_for_last_30_days)] <- median_value
validation$subscribers_for_last_30_days[is.na(validation$subscribers_for_last_30_days)] <- median_value

formula <- as.formula(paste("earnings ~", paste(earnsVars, collapse = " + ")))
suppressWarnings({
  model3 <- glm(formula, data=train, family=binomial(link="logit"))
})

train$pred <- predict(model3, newdata=train, type="response")
test$pred <- predict(model3, newdata=test, type="response")

ggplot(train, aes(x=pred, fill=earnings, group=earnings)) + geom_density(alpha=0.5) +
  theme(text=element_text(size=20))

pretty_perf_table(model3, 
                  train[earnsVars], train[,outcome]==pos,
                  validation[earnsVars], validation[,outcome]==pos,
                  test[earnsVars], test[,outcome]==pos)

pred_train_roc2 <- predict(model3, newdata=train)
pred_val_roc2 <- predict(model3, newdata=validation)

p3 <- plot_roc(pred_train_roc2, train[[outcome]], pred_val_roc2, validation[[outcome]])

formula2 <- as.formula(paste("earnings ~", paste(tVars, collapse = " + ")))
suppressWarnings({
  model3_2 <- glm(formula2, data=train, family=binomial(link="logit"))
  
  pred_train_roc2 <- predict(model3_2, newdata=train)
  pred_val_roc2 <- predict(model3_2, newdata=validation)
  
  p4 <- plot_roc2(pred_train_roc2, train[[outcome]], pred_val_roc2, validation[[outcome]])
  p4
})

df_sum <- gy_data %>% dplyr::select("Urban_population", "subscribers", "Gross.tertiary.education.enrollment....", "channel_type")
names(df_sum)[names(df_sum) == "Gross.tertiary.education.enrollment...."] <- "GTEE"
df_sum <- df_sum %>% drop_na() 
summary(df_sum)

df_sum <- df_sum %>%
  group_by(channel_type) %>%
  summarise(
    mean_urbanpop = mean(Urban_population),
    mean_subs = mean(subscribers),
    mean_gtee = mean(GTEE)
  )
summary(df_sum)

df_to_cluster<- df_sum %>% dplyr::select(contains("mean"))
d <- dist(df_to_cluster, method="euclidean")
pfit <- hclust(d, method="ward.D2")

plot(pfit, cex=0.6,labels=df_sum$channel_type,main="Dendrogram")
rect.hclust(pfit, k=4) # k=5 means we want rectangles to be put around 5 clusters

library(fpc)
kbest.p <- 4
cboot.hclust <- clusterboot(df_to_cluster, clustermethod=hclustCBI,
                            method="ward.D2", k=kbest.p)
summary(cboot.hclust$result)
groups.cboot <- as.factor(cboot.hclust$result$partition)
values<- 1-cboot.hclust$bootbrd/100

sqr_euDist <- function(x, y) {
  sum((x - y)^2)
}
# Function to calculate WSS of a cluster, represented as a n-by-d matrix
# (where n and d are the numbers of rows and columns of the matrix)
# which contains only points of the cluster.
wss <- function(clustermat) {
  c0 <- colMeans(clustermat)
  sum(apply( clustermat, 1, FUN=function(row) {sqr_euDist(row, c0)} ))
}
# Function to calculate the total WSS. Argument `scaled_df`: data frame
# with normalised numerical columns. Argument `labels`: vector containing
# the cluster ID (starting at 1) for each row of the data frame.
wss_total <- function(scaled_df, labels) {
  wss.sum <- 0
  k <- length(unique(labels))
  for (i in 1:k)
    wss.sum <- wss.sum + wss(subset(scaled_df, labels == i))
  wss.sum
}

# Function to calculate total sum of squared (TSS) distance of data
# points about the (global) mean. This is the same as WSS when the
# number of clusters (k) is 1.
tss <- function(scaled_df) {
  wss(scaled_df)
}

CH_index <- function(scaled_df, kmax, method="kmeans") {
  if (!(method %in% c("kmeans", "hclust")))
    stop("method must be one of c('kmeans', 'hclust')")
  npts <- nrow(scaled_df)
  wss.value <- numeric(kmax) # create a vector of numeric type
  # wss.value[1] stores the WSS value for k=1 (when all the
  # data points form 1 large cluster).
  wss.value[1] <- wss(scaled_df)
  if (method == "kmeans") {
    # kmeans
    for (k in 2:kmax) {
      clustering <- kmeans(df_to_cluster, k, nstart=7, iter.max=100)
      wss.value[k] <- clustering$tot.withinss
    }
  } else {
    # hclust
    d <- dist(df_to_cluster, method="euclidean")
    pfit <- hclust(d, method="ward.D2")
    for (k in 2:kmax) {
      labels <- cutree(pfit, k=k)
      wss.value[k] <- wss_total(df_to_cluster, labels)
    }
  }
  
  bss.value <- tss(df_to_cluster) - wss.value 
  B <- bss.value / (0:(kmax-1)) 
  W <- wss.value / (npts - 1:kmax) 
  data.frame(k = 1:kmax, CH_index = B/W, WSS = wss.value)
}

# calculate the CH criterion
crit.df <- CH_index(df_to_cluster, 7, method="hclust")
fig1 <- ggplot(crit.df, aes(x=k, y=CH_index)) +
  geom_point() + geom_line(colour="red") +
  scale_x_continuous(breaks=1:10, labels=1:10) +
  labs(y="CH index") + theme(text=element_text(size=20))
fig2 <- ggplot(crit.df, aes(x=k, y=WSS), color="blue") +
  geom_point() + geom_line(colour="blue") +
  scale_x_continuous(breaks=1:10, labels=1:10) +
  theme(text=element_text(size=20))
grid.arrange(fig1, fig2, nrow=1)

plot(pfit, labels=df_sum$channel_type, main="Cluster Dendrogram for injured people's work experience in different occupations")
rect.hclust(pfit, k=6)

print_clusters <- function(df, groups, cols_to_print) {
  Ngroups <- max(groups)
  for (i in 1:Ngroups) {
    print(paste("cluster", i))
    print(df[groups == i, cols_to_print])
  }
}

groups <- cutree(pfit, k=4)
cols_to_print <- c("channel_type","mean_urbanpop", "mean_subs", "mean_gtee")

princ <- prcomp(df_to_cluster) # Calculate the principal components of df_to_cluster
nComp <- 2 # focus on the first two principal components
# project df_to_cluster onto the first 2 principal components to form a new
# 2-column data frame.
# Assuming groups is a factor
project2D <- as.data.frame(predict(princ, newdata=df_to_cluster)[,1:nComp])
# combine with `groups` and df$Country to form a 4-column data frame
hclust.project2D <- cbind(project2D, cluster=as.factor(groups))

library('grDevices')
library('ggplot2')
find_convex_hull <- function(proj2Ddf, groups) {
  do.call(rbind,
          lapply(unique(groups),
                 FUN = function(c) {
                   f <- subset(proj2Ddf, cluster==c);
                   f[chull(f),]
                 }
          )
  )
}
hclust.hull <- find_convex_hull(hclust.project2D, groups)

ggplot(hclust.project2D, aes(x=PC1, y=PC2)) +
  geom_point(aes(shape=cluster, color=cluster)) +
  
  geom_polygon(data=hclust.hull, aes(group=cluster, fill=as.factor(cluster)),
               alpha=0.4, linetype=0) + theme(text=element_text(size=20))

kbest.p<- 5
km.clusters<- kmeans(df_to_cluster,kbest.p,nstart = 100,iter.max = 100)
km.clusters$centers

km.clusters$size

groups<- km.clusters$cluster
print_clusters(df_sum,groups,"channel_type")

kmClustering.ch <- kmeansruns(df_to_cluster, krange=1:7, criterion="ch")
kmClustering.asw <- kmeansruns(df_to_cluster, krange=1:7, criterion="asw")

kmCritframe <- data.frame(k=1:7, ch=kmClustering.ch$crit,
                          asw=kmClustering.asw$crit)
fig1 <- ggplot(kmCritframe, aes(x=k, y=ch)) +
  geom_point() + geom_line(colour="red") +
  scale_x_continuous(breaks=1:7, labels=1:7) +
  labs(y="CH index") + theme(text=element_text(size=20))
fig2 <- ggplot(kmCritframe, aes(x=k, y=asw)) +
  geom_point() + geom_line(colour="blue") +
  scale_x_continuous(breaks=1:7, labels=1:7) +
  labs(y="ASW") + theme(text=element_text(size=20))
grid.arrange(fig1, fig2, nrow=1)

fig <- c()
kvalues <- seq(2,5)
for (k in kvalues) {
  groups <- kmeans(df_to_cluster, k, nstart=100, iter.max=100)$cluster
  kmclust.project2D <- cbind(project2D, cluster=as.factor(groups),
                             country=df_sum$channel_type)
  kmclust.hull <- find_convex_hull(kmclust.project2D, groups)
  assign(paste0("fig", k),
         ggplot(kmclust.project2D, aes(x=PC1, y=PC2)) +
           geom_point(aes(shape=cluster, color=cluster)) +
           geom_polygon(data=kmclust.hull, aes(group=cluster, fill=cluster),
                        alpha=0.4, linetype=0) +
           labs(title = sprintf("k = %d", k)) +
           theme(legend.position="none", text=element_text(size=20))
  )
}
grid.arrange(fig2, fig3, fig4, fig5, nrow=2)










# Decision Tree all variables model data
model_data1 <- data.frame(
  model = c("training", "validation", "test"),
  precision = sprintf("%.4f", c(0.9887, 0.9792, 0.9434)),
  recall = sprintf("%.4f", c(1.0000, 1.0000, 0.9901)),
  f1 = sprintf("%.4f", c(0.9943, 0.9895, 0.9662)),
  dev.norm = sprintf("%.4f", c(0.2293, 0.4530, 1.3069))
)

# Decision Tree reprocessed variables model data
model_data2 <- data.frame(
  model = c("training", "validation", "test"),
  precision = sprintf("%.4f", c(0.7825, 0.8393, 0.7214)),
  recall = sprintf("%.4f", c(1.0000, 1.0000, 1.0000)),
  f1 = sprintf("%.4f", c(0.8780, 0.9126, 0.8382)),
  dev.norm = sprintf("%.4f", c(5.561, 4.077, 7.281))
)

# Define the UI
ui <- fluidPage(
  titlePanel("Model Analysis"),
  
  sidebarLayout(
    sidebarPanel(
      selectInput("model_type", "Choose model type:", choices = c("Single-variate model", "Multivariate models", "Clustering")),
      
      conditionalPanel(
        condition = "input.model_type == 'Single-variate model'",
        selectInput("view_attr_single", "Select attribute:", choices = c("ROC curves"))
      ),
      
      conditionalPanel(
        condition = "input.model_type == 'Multivariate models'",
        selectInput("model_choice", "Choose model:", choices = c("Decision Tree", "Logistic Regression")),
        conditionalPanel(
          condition = "input.model_choice == 'Decision Tree'",
          selectInput("variable_choice_dt", "Choose variable type:", choices = c("All Variables", "Reprocessed Variables"))
        ),
        conditionalPanel(
          condition = "input.model_choice == 'Logistic Regression'",
          selectInput("variable_choice_lr", "Choose variable type:", choices = c("All Variables", "Reprocessed Variables"))
        )
      ),
      
      conditionalPanel(
        condition = "input.model_type == 'Clustering'",
        sliderInput("K_select", "Range of K:", min = 2, max = 5, value = 2)
      )
      
    ),
    
    mainPanel(
      plotOutput("roc_plot", height = "400px", width = "600px"),
      div(
        tableOutput("model_data_table"),
        style = "margin-top: 100px;" # 调整外边距的值以满足你的需求
      )
    )
  )
)


# Shiny App Part

# Define the server
server <- function(input, output) {
  
  # Conditional rendering for ROC curve plot
  output$roc_plot <- renderPlot({
    if (input$model_type == "Single-variate model" && input$view_attr_single == "ROC curves") {
      
      # Set up a blank plot with appropriate axes and labels
      plot(NA, NA, xlim = c(0, 1), ylim = c(0, 1), xlab = "False Positive Rate", ylab = "True Positive Rate", main = "Combined ROC Curves")
      
      # Plot ROC curves for the top four variables
      library(ROCR) # Make sure you have ROCR package installed
      colors <- c("red", "green", "blue", "purple")
      for (i in 1:4) {
        v <- top_four_variables[i]
        p <- paste('pred_', v, sep = '')
        perf <- performance(prediction(test[, p], test[, outcome] == pos), "tpr", "fpr")
        plot(perf, col = colors[i], add = TRUE, lwd = 2, lty = i)
        legend(0, 1, legend = top_four_variables, col = colors, lty = 1:4, title = "Variables", cex = 0.8)
      }
    }
    
    if (input$model_type == "Multivariate models" && input$model_choice == "Decision Tree" && input$variable_choice_dt == "All Variables") {
      # Set up a blank plot with appropriate axes and labels
      plot(NA, NA, xlim = c(0, 1), ylim = c(0, 1), xlab = "False Positive Rate", ylab = "True Positive Rate", main = "Decision Tree ROC Curve")
      
      # Plot ROC curves for Decision Tree
      pred_train_roc <- predict(tmodel, newdata=train)
      pred_val_roc <- predict(tmodel, newdata=validation)
      plot_roc(pred_train_roc, train[[outcome]], pred_val_roc, validation[[outcome]])
      
    }
    
    if (input$model_type == "Multivariate models" && input$model_choice == "Decision Tree" && input$variable_choice_dt == "Reprocessed Variables") {
      # Set up a blank plot with appropriate axes and labels
      plot(NA, NA, xlim = c(0, 1), ylim = c(0, 1), xlab = "False Positive Rate", ylab = "True Positive Rate", main = "Decision Tree ROC Curve")
      
      # Plot ROC curves for Decision Tree
      pred_train_roc <- predict(tmodel2, newdata=train)
      pred_val_roc <- predict(tmodel2, newdata=validation)
      plot_roc2(pred_train_roc, train[[outcome]], pred_val_roc, validation[[outcome]])
    }
    
    if (input$model_type == "Multivariate models" && input$model_choice == "Logistic Regression" && input$variable_choice_dt == "All Variables") {
      # Set up a blank plot with appropriate axes and labels
      plot(NA, NA, xlim = c(0, 1), ylim = c(0, 1), xlab = "False Positive Rate", ylab = "True Positive Rate", main = "Logistic Regression ROC Curve")
      
      # Plot ROC curves for Decision Tree
      pred_train_roc2 <- predict(model3, newdata=train)
      pred_val_roc2 <- predict(model3, newdata=validation)
      plot_roc(pred_train_roc2, train[[outcome]], pred_val_roc2, validation[[outcome]])
    }
    
    if (input$model_type == "Multivariate models" && input$model_choice == "Logistic Regression" && input$variable_choice_dt == "Reprocessed Variables") {
      # Set up a blank plot with appropriate axes and labels
      plot(NA, NA, xlim = c(0, 1), ylim = c(0, 1), xlab = "False Positive Rate", ylab = "True Positive Rate", main = "Logistic Regression ROC Curve")
      
      # Plot ROC curves for Decision Tree
      pred_train_roc2 <- predict(model3_2, newdata=train)
      pred_val_roc2 <- predict(model3_2, newdata=validation)
      plot_roc2(pred_train_roc2, train[[outcome]], pred_val_roc2, validation[[outcome]])
    }
    
    if (input$model_type == "Clustering" && input$K_select == 2) {
      fig <- ggplot(kmclust.project2D, aes(x=PC1, y=PC2)) +
        geom_point(aes(shape=cluster, color=cluster)) +
        geom_polygon(data=kmclust.hull, aes(group=cluster, fill=cluster),
                     alpha=0.4, linetype=0) +
        labs(title = sprintf("k = 2")) +
        theme(legend.position="none", text=element_text(size=20))
      
      print(fig2)
    }
    
    if (input$model_type == "Clustering" && input$K_select == 3) {
      fig <- ggplot(kmclust.project2D, aes(x=PC1, y=PC2)) +
        geom_point(aes(shape=cluster, color=cluster)) +
        geom_polygon(data=kmclust.hull, aes(group=cluster, fill=cluster),
                     alpha=0.4, linetype=0) +
        labs(title = sprintf("k = 3")) +
        theme(legend.position="none", text=element_text(size=20))
      
      print(fig3)
    }
    
    if (input$model_type == "Clustering" && input$K_select == 4) {
      fig <- ggplot(kmclust.project2D, aes(x=PC1, y=PC2)) +
        geom_point(aes(shape=cluster, color=cluster)) +
        geom_polygon(data=kmclust.hull, aes(group=cluster, fill=cluster),
                     alpha=0.4, linetype=0) +
        labs(title = sprintf("k = 4")) +
        theme(legend.position="none", text=element_text(size=20))
      
      print(fig4)
    }
    
    if (input$model_type == "Clustering" && input$K_select == 5) {
      fig <- ggplot(kmclust.project2D, aes(x=PC1, y=PC2)) +
        geom_point(aes(shape=cluster, color=cluster)) +
        geom_polygon(data=kmclust.hull, aes(group=cluster, fill=cluster),
                     alpha=0.4, linetype=0) +
        labs(title = sprintf("k = 5")) +
        theme(legend.position="none", text=element_text(size=20))
      
      print(fig5)
    }
    
    
  }, height = 500, width = 800)
  
  
  output$model_data_table <- renderTable({
    if (input$model_type == "Multivariate models" && input$model_choice == "Decision Tree" && input$variable_choice_dt == "All Variables") {
      return(model_data1)
    }
    
    if (input$model_type == "Multivariate models" && input$model_choice == "Decision Tree" && input$variable_choice_dt == "Reprocessed Variables") {
      return(model_data2)
    }
  })
  
}

# Run the app
shinyApp(ui, server)

