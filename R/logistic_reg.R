### LOGISTIC REGRESSION (one-vs-all) classifier for color recognition project

# The colour pictures are characterized by a 10 x 10 HUE values
# matrix that is stored in the rows. The labels
# corresponding to each digit is 1-blue 2-red 3-green

graphics.off() # close all plots
rm(list=ls()) # remove all objects

### IMPORTING THE DATA

library("rjson")
data <- fromJSON(file = "final_data.json")

### DATA PROCESSING
# Extracting the number of blue, red and green samples.
# N[1]: blues, N[2]: reds, N[3]: greens
N <- unname(sapply(data[[1]],'[[',2))

# Generating the matrix samples X, vector Y 
X <- c()
for (i in 1:length(data[[2]])){
  X <- rbind(X, as.vector(apply(as.matrix(data[[2]][[i]][[4]]), 1, function(x) unlist(x)))) #ASK!!!! t(M) if by row
}

y <- c(rep(1,N[1]),rep(2,N[2]),rep(3,N[3])) #1 - blue, 2 - red, 3 - green

# data characteristics
n = dim(X)[1] #number of pictures
q = dim(X)[2] #number of HUE values per picture

#Splitting the data in training and testing data

set.seed(12345) #the same used in Neaural Networks
train_p <- 0.9 #the same used in Neaural Networks
TrainingSize <- floor(train_p*nrow(X))
SelectRow <- sample(seq_len(nrow(X)), size = TrainingSize) 

X_train = X[c(SelectRow),]
y_train = y[c(SelectRow)]
X_test = X[-c(SelectRow),]
y_test = y[-c(SelectRow)]

# training data characteristics
n_train = dim(X_train)[1]
q_train = dim(X_train)[2]

### ONE-VS-ALL LOGISTIC REGRESSION: logistic classifier for the 3 colours. 

#creating empty matrix
num_labels = 3
beta_one_vs_all = matrix(0,q_train + 1, num_labels) #betas for each colour in seperate columns


for (c in 1:num_labels) {
  #creating one-vs-all data (with values 0 and 1), where the color in question has the value of 1 and all other colors - 0.
  id_selected=which(y_train==c)
  y_c = y_train
  y_c[-id_selected] = 0  
  y_c[id_selected] = 1

  #running logistic regression for each color
  data = data.frame(y_c,X_train)
  model_glmfit_c = glm(y_c ~., data, start =rep(0,q_train+1) ,family=binomial(link="logit"),
                       control=list(maxit = 100, trace = FALSE) )
  beta_glmfit_c  = model_glmfit_c$coefficients # coefficients
  beta_glmfit_c[is.na(beta_glmfit_c)]=0
  
beta_one_vs_all[, c] = beta_glmfit_c # beta coefficients for each color in seperate columns
}

### Classification, Missclassification matrix and Errors
#Training data:
y_classified = apply( cbind(rep(1,n_train), X_train) %*% beta_one_vs_all , 1, FUN=which.max) 
Empirical_error_one_vs_all = length(which(y_classified != y_train)) / n_train
Accuracy <- 1 - Empirical_error_one_vs_all

misclassification_matrix = matrix(0,num_labels, num_labels)
for (i in 1:num_labels) {
  for (j in 1:num_labels) {
    misclassification_matrix[i, j] = length(which((y_train == i) & (y_classified == j))) / length(which((y_train == i)))
  }
}


#Testing data:
n_test = dim(X_test)[1]
y_classified_test = apply( cbind(rep(1,n_test), X_test) %*% beta_one_vs_all , 1, FUN=which.max)
Empirical_error_one_vs_all_test = length(which(y_classified_test != y_test)) / n_test
Accuracy_test <- 1 - Empirical_error_one_vs_all_test

misclassification_matrix_test = matrix(0,num_labels, num_labels)
for (l in 1:num_labels) {
  for (k in 1:num_labels) {
    misclassification_matrix_test[l, k] = length(which((y_test == l) & (y_classified_test == k))) / length(which((y_test == l)))
  }
}

##################################################################################### 
###LOGISTIC REGRESSION (one-vs-all) classifier for color recognition project 
###WITH INDEPENDENT VARIABLES ADJUSTMENTS (Aggregation)

#The results in previous part depends a lot on the picture colors' place. If the picture has
#its key colors in the corner only, the algorithm will be weak to predict it. In order to deal with
#it, we run regression on aggregated data.

meanexcludingzeros <- function(inputdata) {
  #Calculate means excluding the zeros for the data matrix
  if (all(inputdata==0)) 0 else mean(inputdata[inputdata!=0])
}

X_Means = apply(X,1,meanexcludingzeros) #averages of HUE values per picture
#Since the HUE values are very dispersed (there are a lot of observations in both ends of HUE value spectrum),
#the mean is very sensitive, especially 
#for the red color. Median did not perform better. 

MostCommonValue <- function(inputdata) {
  #Find the most common HUE value excluding the noise (HUE numbers of white colour (0)) for the data matrix
  if (all(inputdata==0)) 0 else as.numeric(names(which.max(table(inputdata[inputdata!=0]))))
}

X_MostCommon = apply(X,1,MostCommonValue) #Most common HUE value per picture

# Boxplot of Mean and most common HUE values per picture

y_text = y 
y_text[y_text == 1] <- 'Blue' #Changing numerical values to strings
y_text[y_text == 2] <- 'Red' #Changing numerical values to strings
y_text[y_text == 3] <- 'Green' #Changing numerical values to strings


par(mfrow=c(1,2))
boxplot(X_Means~y_text,  ylab="a) Mean", col=c('Blue', 'Green','Red'))

boxplot(X_MostCommon~y_text,  ylab="b) Most Common", col=c('Blue', 'Green','Red'))
mtext("HUE values characteristics across pictures for blue, red and green colors", side = 3, line = -2.5, outer = TRUE)

#Splitting the data in training and testing data

X_MostCommon_train = X_MostCommon[c(SelectRow)]
X_MostCommon_test = X_MostCommon[-c(SelectRow)]

q_train_alt = 1 #aggregate HUE values (by most common) leads to 1 explanatory variable

### ONE-VS-ALL LOGISTIC REGRESSION: logistic classifier for the 3 colours using THE MOST COMMON VALUE. 

#creating empty matrix

beta_one_vs_all_MostCommon = matrix(0,q_train_alt + 1, num_labels)

for (c in 1:num_labels) {
  #creating one-vs-all data (with values 0 and 1), where the color in question has the value of 1 and all other colors - 0.
  id_selected=which(y_train==c)
  y_c = y_train
  y_c[-id_selected] = 0  
  y_c[id_selected] = 1
  
  #running logistic regression for each color
  data = data.frame(y_c,X_MostCommon_train)
  model_glmfit_c = glm(y_c ~., data, start =rep(0,q_train_alt+1) ,family=binomial(link="logit"),
                       control=list(maxit = 100, trace = FALSE) )
  beta_glmfit_c  = model_glmfit_c$coefficients # coefficients
  beta_glmfit_c[is.na(beta_glmfit_c)]=0
  
  beta_one_vs_all_MostCommon[, c] = beta_glmfit_c # beta coefficients for each color in seperate columns
}

### Classification, Missclassification matrix and Errors
#Training data:
y_MostCommon_classified = apply( cbind(rep(1,n_train), X_MostCommon_train) %*% beta_one_vs_all_MostCommon , 1, FUN=which.max) 
Empirical_error_one_vs_all_MostCommon = length(which(y_MostCommon_classified != y_train)) / n_train
Accuracy_MostCommon <- 1 - Empirical_error_one_vs_all_MostCommon

misclassification_matrix_MostCommon = matrix(0, num_labels, num_labels)
for (i in 1:num_labels) {
  for (j in 1:num_labels) {
    misclassification_matrix_MostCommon[i, j] = length(which((y_train == i) & (y_MostCommon_classified == j))) / length(which((y_train == i)))
  }
}

#Testing data:
n_test = dim(X_test)[1]
y_MostCommon_classified_test = apply(cbind(rep(1,n_test), X_MostCommon_test) %*% beta_one_vs_all_MostCommon , 1, FUN=which.max)
Empirical_error_one_vs_all_test_MostCommon = length(which(y_MostCommon_classified_test != y_test)) / n_test
Accuracy_MostCommon_test <- 1 - Empirical_error_one_vs_all_test_MostCommon

misclassification_matrix_test_MostCommon = matrix(0,num_labels, num_labels)
for (l in 1:num_labels) {
  for (k in 1:num_labels) {
    misclassification_matrix_test_MostCommon[l, k] = length(which((y_test == l) & (y_MostCommon_classified_test == k))) / length(which((y_test == l)))
  }
}
