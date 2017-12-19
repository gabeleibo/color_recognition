### IMPORTING THE DATA
#install.packages("rjson")
library("rjson")
data <- fromJSON(file = "final_data.json")

### DATA PROCESSING
# Extracting the number of blue, red and green samples.
# N[1]: blues, N[2]: reds, N[3]: greens
N <- unname(sapply(data[[1]], '[[', 2))

# Generating the matrix samples X, vector Y (where 0: blue, 1: red, 2: green)
X <- c()
for (i in 1:length(data[[2]])){
  X <- rbind(X, as.vector(apply(as.matrix(data[[2]][[i]][[4]]), 1, function(x) unlist(x))))
}

Y <- matrix(c(rep(c(0,0,1), N[1]), rep(c(0,1,0), N[2]), rep(c(1,0,0), N[3])),
            ncol = 3, byrow = TRUE)

# Pretreatment of the data
# [sigmamin,sigmamax] this is [0,1] for the logistic function
sigma_min <- 0
sigma_max <- 1

X_max <- apply(X, 2, max)
X_min <- apply(X, 2, min)
X_treat <- (sigma_max - sigma_min) / (X_max - X_min) * (X - X_min) + sigma_min


###############################################################################

### NEURONAL NETWORK
# Defining the activation function
sigmoid <- function(z){
  return(1 / (1 + exp(-z)))
}

## FORWARD PROPAGATION
forward <- function(X, weights, size_hidden, size_input, size_output){
  # Weights matrices
  Theta1 <- matrix(weights[1:(size_hidden * (size_input + 1))], nrow = size_hidden)
  Theta2 <- matrix(weights[((size_hidden * (size_input + 1)) + 1):length(weights)],
                   nrow = size_output)
  
  # Including intercept in X and transposing the matrix
  n <- nrow(X)
  X <- cbind(rep(1, n), X)
  
  # Computaring neurons values and output
  l1 <- sigmoid(X %*% t(Theta1))
  l1_i <- cbind(rep(1, n), l1)
  l2 <- sigmoid(l1_i %*% t(Theta2))
  
  # Classifying
  M <- apply(l2, 1, max)
  Y_classified <- floor(l2 / M)
  
  return(list(Theta1 = Theta1, Theta2 = Theta2, l1 = l1, l2 = l2,
              Y_classified = Y_classified))
}


## BACKWARD PROPAGATION
# Neuronal network cost function
nnminuslogLikelihood <- function(weights, size_input, size_hidden, size_output, X,
                                 Y, lambda){
  # Computes the neuronal network cost function for a three layer
  # neural network which performs classification using backpropagation.
  #
  # Args:
  #   x: Vector with all the weights from the NN.
  #   size_input: Number of neurons of the first layer or of inputs.
  #   size_hidden: Number of neurons of the hidden layer.
  #   size_output: Number of neurons of the last layer or of outputs.
  #   X: Matrix of explanatory data.
  #   Y: Matrix of explained data.
  #   lamba : Relaxation.
  #
  # Returns:
  #   The -logLikelihood or cost
  
  n <- dim(X)[1]
  
  # Forward propagation step
  forward_result <- forward(X, weights, size_hidden, size_input, size_output)
  Theta1 <- forward_result$Theta1
  Theta2 <- forward_result$Theta2
  l2 <- forward_result$l2 
  
  # Computing the logLikelihood
  J <- 0
  J <- apply(-Y * log(l2), 1, sum) - apply((1 - Y) * log(1 - l2), 1, sum)
  J <- sum(J) / n + (lambda /(2 * n)) * (sum(Theta1[, 2:dim(Theta1)[2]]^2) + sum(Theta2[, 2:dim(Theta2)[2]]^2))
  
  return(J)
}

# Neuronal network gradient of the cost function
nnminuslogLikelihood_grad <- function(weights, size_input, size_hidden, size_output, X,
                                      Y, lambda){
  # Computes the neuronal network gradient of the cost function for a three layer
  # neural network which performs classification using backpropagation.
  #
  # Args:
  #   x: Vector with all the weights from the NN.
  #   size_input: Number of neurons of the first layer or of inputs.
  #   size_hidden: Number of neurons of the hidden layer.
  #   size_output: Number of neurons of the last layer or of outputs.
  #   X: Matrix of explanatory data.
  #   Y: Matrix of explained data.
  #   lamba : Relaxation.
  #
  # Returns:
  #   The gradient
  
  n <- dim(X)[1]
  
  # Forward propagation step
  forward_result <- forward(X, weights, size_hidden, size_input, size_output)
  Theta1 <- forward_result$Theta1
  Theta2 <- forward_result$Theta2
  l1 <- forward_result$l1
  l2 <- forward_result$l2 
  
  # Computing the deltas and Deltas
  delta3 <- t(l2 - Y)
  Delta2 <- delta3 %*% cbind(rep(1, n), l1)
  delta2 <- (t(Theta2) %*% delta3)[2:(size_hidden + 1), ] * (t(l1) * t(1 - l1))
  Delta1 <- delta2 %*% cbind(rep(1, n), X)
  
  # Computing the gradient of each group of weights
  Theta1_grad <- Delta1 / n
  Theta1_grad[, 2:dim(Theta1_grad)[2]] <- 
    Theta1_grad[, 2:dim(Theta1_grad)[2]] + (lambda / n) * Theta1[, 2:dim(Theta1_grad)[2]]
  Theta2_grad <- Delta2 / n
  Theta2_grad[, 2:dim(Theta2_grad)[2]] <- 
    Theta2_grad[, 2:dim(Theta2_grad)[2]] + (lambda / n) * Theta2[, 2:dim(Theta2_grad)[2]]
  
  return(c(as.vector(Theta1_grad), as.vector(Theta2_grad)))
}

# Optimization of the weights
p <- ncol(X)    # size inputs
k <- length(N)  # size labels
set.seed(12345)

# Matrices with information about the MSEs of training and testing and the weights for
# different runs in the size of the training data and number of neurons.
MSE_Training <- matrix(0, nrow = 46, ncol = 7)
MSE_Testing <- matrix(0, nrow = 46, ncol = 7)

# Loop for  size of the training and testing data
for (j in 1:7){
  # Determining the training and testing data
  per <- 0.3 + 0.1 * j
  SelectRow <- c(sample(seq_len(N[1]), size = floor(per * N[1])),
                 sample((seq_len(N[2]) + N[1]), size = floor(per * N[2])),
                 sample((seq_len(N[3]) + N[1] + N[2]), size = floor(per * N[3])))
  TrainingData <- X_treat[SelectRow, ]
  TrainingOutput <- Y[SelectRow, ]
  ValidationData <- X_treat[-SelectRow, ]
  ValidationOutput <- Y[-SelectRow, ]
  
  # Loop for number of neurons in the hidden layer
  for (i in 5:50){
    Theta1 <- matrix(runif(i * (p + 1)), nrow = i)
    ThetaF <- matrix(runif(k* (i + 1)), nrow = k)
    weights <- c(as.vector(Theta1), as.vector(ThetaF))
    
    # Optimization k
    #options <- list(trace = 1, iter.max = 100) # print every iteration 
    backp_result <- nlminb(weights, 
                           objective   = nnminuslogLikelihood,
                           gradient    = nnminuslogLikelihood_grad,
                           hessian     = NULL,
                           size_input  = p,
                           size_hidden = i,
                           size_output = k,
                           X           = TrainingData,
                           Y           = TrainingOutput,
                           lambda      = 1)
                          #control = options)
    
    # Getting the weights and saving them in the weights matrix
    Weights_backp <- backp_result$par
  
    # Computing the MSEs and misclassification ratio of training and testing. 
    # Saving them in the respective matrix. 
    Y_Train <- forward(TrainingData, Weights_backp, i, p, k)$l2
    Y_Test <- forward(ValidationData, Weights_backp, i, p, k)$l2
    MSE_Training[(i - 4), j] <- sum((Y_Train - TrainingOutput)^2) / nrow(TrainingOutput)
    MSE_Testing[(i - 4), j] <- sum((Y_Test - ValidationOutput)^2) / nrow(ValidationOutput)
  }
  
  write.csv(MSE_Training, "train_error.csv")
  write.csv(MSE_Testing, "test_error.csv")
}


## PERFORMANCE ANALYSIS
# Selecting the best performace neuronal network and training data size
# MSE_Training <- as.matrix(read.csv("train_error.csv"))[,2:8]
# MSE_Testing <- as.matrix(read.csv("test_error.csv"))[,2:8]
averagePerforming <- (apply(MSE_Testing, 2, sum) / nrow(MSE_Testing))[1:(ncol(MSE_Testing) - 1)]
j <- which(averagePerforming == min(averagePerforming))
per <- j * 0.1 + 0.3
i <- which(MSE_Testing[, j] == min(MSE_Testing[, j])) + 4 

# Plotting the MSEs per number of neurons in the hidden layer -between 5 and 50 neurons-
# for size of training data with better results  
Hidden_layer <- 5:(length(MSE_Testing[, ((per - 0.3) / 0.1)]) + 4)
plot(Hidden_layer, MSE_Testing[, ((per - 0.3) / 0.1)], type = "l", col = "red", ylim=c(0,0.5))
lines(Hidden_layer, MSE_Training[, ((per - 0.3) / 0.1)], col="black")

# Getting the weights
SelectRow <- c(sample(seq_len(N[1]), size = floor(per * N[1])),
               sample((seq_len(N[2]) + N[1]), size = floor(per * N[2])),
               sample((seq_len(N[3]) + N[1] + N[2]), size = floor(per * N[3])))
TrainingData <- X_treat[SelectRow, ]
TrainingOutput <- Y[SelectRow, ] 
ValidationData <- X_treat[-SelectRow, ]
ValidationOutput <- Y[-SelectRow, ]

Theta1 <- matrix(runif(i * (p + 1)), nrow = i)
ThetaF <- matrix(runif(k* (i + 1)), nrow = k)
weights <- c(as.vector(Theta1), as.vector(ThetaF))

#options <- list(trace = 1, iter.max = 100) # print every iteration 
backp_result <- nlminb(weights, 
                       objective   = nnminuslogLikelihood,
                       gradient    = nnminuslogLikelihood_grad,
                       hessian     = NULL,
                       size_input  = p,
                       size_hidden = i,
                       size_output = k,
                       X           = TrainingData,
                       Y           = TrainingOutput,
                       lambda      = 1)
                      #control = options)

# Getting the weights and saving them in the weights matrix
Weights_backp <- backp_result$par
#save(Weights_backp, file = "weights.rda")

# Creating the confusion matrix for the testing sample with size 0.1
# of total sample and with k number of neurons in the hidden layer.
Y_classified <- forward(ValidationData, Weights_backp, i , p, k)$Y_classified

Testing_actual <- max.col(ValidationOutput)
Testing_actual[which(Testing_actual == 3)] <- "blue"
Testing_actual[which(Testing_actual == 2)] <- "red"
Testing_actual[which(Testing_actual == 1)] <- "green"

Testing_predicted <- max.col(Y_classified)
Testing_predicted[which(Testing_predicted == 3)] <- "blue"
Testing_predicted[which(Testing_predicted == 2)] <- "red"
Testing_predicted[which(Testing_predicted == 1)] <- "green"

table <- table(Testing_actual, Testing_predicted)
table

# Measures of Performance
# Percentage of the correctly classified predictions over all
accuracy <- sum(diag(table)) / sum(table)
# Fraction of correctly predicted of an actual class
precision <- diag(table) / apply(table, 2, sum)
# Fraction of correctly predicted of a predicted class
recall <- diag(table) / apply(table, 1, sum)
# Weighted average of precision and recall
f1 <- 2 * precision * recall / (precision + recall)