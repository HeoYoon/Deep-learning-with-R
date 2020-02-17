
sigmoid <- function(Z) {
  A <- 1/(1 + exp(-Z))
  cache <- Z
  return(list(A = A, Z = Z))
}

relu <- function(Z) {
  A <- pmax(Z, 0)
  cache <- Z
  return(list(A = A, Z = Z))
}

tanh <- function(Z) {
  A <- sinh(Z)/cosh(Z)
  cache <- Z
  return(list(A = A, Z = Z))
}

softmax <- function(Z) {
  # get unnormalized probabilities
  exp_scores = exp(t(Z))
  # get the normalized probabilities
  A = exp_scores/rowSums(exp_scores)
  return(list(A = A, Z = Z))
}

derivative_sigmoid <- function(dA, cache) {
  Z <- cache
  s <- 1/(1 + exp(-Z))
  dZ <- dA * s * (1 - s)
  return(dZ)
}

derivative_relu <- function(dA, cache) {
  Z <- cache
  dZ <- dA
  a <- (Z > 0) # Find which values of Z are greater than zero
  dZ <- dZ * a # when Z <= 0, dZ is set to zero
  return(dZ)
}

derivative_tanh <- function(dA, cache) {
  Z = cache
  a = sinh(Z) / cosh(Z)
  dZ = dA * (1 - a^2)
  return(dZ)
}

derivative_softmax <- function(dA, cache, X, Y, num_classes) {
  y.mat <- matrix(Y, ncol = 1)
  y <- matrix(0, nrow = length(Y), ncol = num_classes)
  for (i in 0:(num_classes - 1)) {
    y[y.mat[, 1] == i, i + 1] <- 1
  }
  Z <- cache
  exp_scores = exp(t(Z))
  probs = exp_scores/rowSums(exp_scores)
  dZ = probs - y
  return(dZ)
}


initialize_params <- function(layers_dims, initialization){
  set.seed(2)
  layerParams <- list()
  for(layer in 2:length(layers_dims)){
    if(initialization == 'zero'){
      n = 0 * rnorm(layers_dims[layer] * layers_dims[layer - 1])
      layerParams[[paste('W', layer - 1, sep = "")]] =
        matrix(n,
               nrow = layers_dims[layer],
               ncol = layers_dims[layer - 1])
      layerParams[[paste('b', layer - 1, sep = "")]] =
        matrix(rep(0, layers_dims[layer]),
               nrow = layers_dims[layer],
               ncol = 1)
    }
    else if(initialization == 'random'){
      n = rnorm(layers_dims[layer] * layers_dims[layer - 1],
                mean = 0,
                sd = 1) * 0.01
      layerParams[[paste('W', layer - 1, sep = "")]] =
        matrix(n,
               nrow = layers_dims[layer],
               ncol = layers_dims[layer - 1])
      layerParams[[paste('b', layer - 1, sep = "")]] =
        matrix(rep(0, layers_dims[layer]),
               nrow = layers_dims[layer],
               ncol = 1)
    }
    else if(initialization == 'He'){
      n = rnorm(layers_dims[layer] * layers_dims[layer - 1], mean = 0, sd = 1) *
        sqrt(2/layers_dims[layer - 1])
      layerParams[[paste('W',layer - 1, sep = "")]] =
        matrix(n,
               nrow = layers_dims[layer],
               ncol = layers_dims[layer - 1])
      layerParams[[paste('b',layer - 1,sep = "")]] =
        matrix(rep(0, layers_dims[layer]),
               nrow = layers_dims[layer],
               ncol = 1)
    }
    else if(initialization == 'Xavier'){
      n = rnorm(layers_dims[layer] * layers_dims[layer - 1], mean = 0, sd = 1) *
        sqrt(1 / layers_dims[layer - 1])
      layerParams[[paste('W',layer - 1, sep = "")]] =
        matrix(n,
               nrow = layers_dims[layer],
               ncol = layers_dims[layer - 1])
      layerParams[[paste('b',layer - 1, sep = "")]] =
        matrix(rep(0, layers_dims[layer]),
               nrow = layers_dims[layer],
               ncol = 1)
    }
  }
  return(layerParams)
}

f_prop_helper <- function(A_prev, W, b, hidden_layer_act){
  Z <-sweep(W %*% A_prev, 1, b, '+')
  forward_cache <- list("A_prev" = A_prev, "W" = W, "b" = b)
  if(hidden_layer_act == "sigmoid"){
    act_values = sigmoid(Z)
  }
  else if (hidden_layer_act == "relu"){
    act_values = relu(Z)
  }
  else if(hidden_layer_act == 'tanh'){
    act_values = tanh(Z)
  }
  else if(hidden_layer_act == 'softmax'){
    act_values = softmax(Z)
  }
  cache <- list("forward_cache" = forward_cache,
                "activation_cache" = act_values[['Z']])
  return(list("A" = act_values[['A']], "cache" = cache))
}

forward_prop <- function(X, parameters, hidden_layer_act, output_layer_act) {
  caches <- list()
  A <- X
  L <- length(parameters)/2
  # Loop through from layer 1 to upto layer L-1
  for (l in 1:(L - 1)) {
    A_prev <- A
    W <- parameters[[paste("W", l, sep = "")]]
    b <- parameters[[paste("b", l, sep = "")]]
    actForward <- f_prop_helper(A_prev, W, b, hidden_layer_act[[l]])
    A <- actForward[["A"]]
    caches[[l]] <- actForward
  }
  W <- parameters[[paste("W", L, sep = "")]]
  b <- parameters[[paste("b", L, sep = "")]]
  actForward = f_prop_helper(A, W, b, output_layer_act)
  AL <- actForward[["A"]]
  caches[[L]] <- actForward
  return(list(AL = AL, caches = caches))
}

compute_cost <- function(AL, X, Y, num_classes, output_layer_act){
  if(output_layer_act == "sigmoid"){
    m = length(Y)
    cross_entropy_cost = -(sum(Y * log(AL) + (1 - Y) * log(1 - AL))) / m
  }
  else if(output_layer_act == "softmax"){
    m = ncol(X)
    y.mat <- matrix(Y, ncol = 1)
    y <- matrix(0, nrow = m, ncol = num_classes)
    for (i in 0:(num_classes - 1)) {
      y[y.mat[, 1] == i, i+1] <- 1
    }
    correct_logprobs <- -log(AL)
    cross_entropy_cost <- sum(correct_logprobs * y) / m
  }
  return(cross_entropy_cost)
}

back_prop_helper <- function(dA, cache, Y, hidden_layer_act, num_classes){
  forward_cache <-cache[['forward_cache']]
  # Get Z
  activation_cache <- cache[['activation_cache']]
  A_prev <- forward_cache[['A_prev']]
  m = dim(A_prev)[2]
  if(hidden_layer_act == "relu"){
    dZ <- derivative_relu(dA, activation_cache)
  }
  else if(hidden_layer_act == "sigmoid"){
    dZ <- derivative_sigmoid(dA, activation_cache)
  }
  else if(hidden_layer_act == "tanh"){
    dZ <- derivative_tanh(dA, activation_cache)
  }
  else if(hidden_layer_act == "softmax"){
    dZ <- derivative_softmax(dAL, activation_cache, X, Y, num_classes)
  }
  W <- forward_cache[['W']]
  b <- forward_cache[['b']]
  m = dim(A_prev)[2]
  if(hidden_layer_act == 'softmax'){
    dW = 1 / m * t(dZ) %*% t(A_prev) #+ (lambd / m) * W
    db = 1 / m * colSums(dZ)
    dA_prev = t(W) %*% t(dZ)
  }
  else{
    dW = 1 / m * dZ %*% t(A_prev)
    db = 1 / m * rowSums(dZ)
    dA_prev = t(W) %*% dZ
  }
  return(list("dA_prev" = dA_prev, "dW" = dW, "db" = db))
}

back_prop <- function(AL,
                      Y,
                      caches,
                      hidden_layer_act,
                      output_layer_act,
                      num_classes){
  gradients = list()
  L = length(caches)
  m = dim(AL)[2]
  if(output_layer_act == "sigmoid"){
    dAL = -((Y/AL) - (1 - Y)/(1 - AL))
  }
  else if(output_layer_act == 'softmax') {
    y.mat <- matrix(Y, ncol = 1)
    y <- matrix(0, nrow = length(Y), ncol = num_classes)
    for (i in 0:(num_classes - 1)) {
      y[y.mat[, 1] == i, i + 1] <- 1
    }
    dAL = (AL - y)
  }
  current_cache = caches[[L]]$cache
  loop_back_vals <- back_prop_helper(dAL, current_cache,
                                     Y,
                                     hidden_layer_act = output_layer_act,
                                     num_classes)
  gradients[[paste("dA", L, sep = "")]] <- loop_back_vals[['dA_prev']]
  gradients[[paste("dW", L, sep = "")]] <- loop_back_vals[['dW']]
  gradients[[paste("db", L, sep = "")]] <- loop_back_vals[['db']]
  for(l in (L - 1):1){
    current_cache = caches[[l]]$cache
    loop_back_vals = back_prop_helper(gradients[[paste('dA', l + 1, sep = "")]],
                                      current_cache,
                                      Y,
                                      hidden_layer_act[[l]],
                                      num_classes)
    gradients[[paste("dA", l, sep = "")]] <- loop_back_vals[['dA_prev']]
    gradients[[paste("dW", l, sep = "")]] <- loop_back_vals[['dW']]
    gradients[[paste("db", l, sep = "")]] <- loop_back_vals[['db']]
  }
  return(gradients)
}

update_params <- function(parameters, gradients, learning_rate) {
  L = length(parameters)/2
  for (l in 1:L) {
    parameters[[paste("W", l, sep = "")]] = parameters[[paste("W",
                                                              l, sep = "")]] - learning_rate * gradients[[paste("dW",
                                                                                                                l, sep = "")]]
    parameters[[paste("b", l, sep = "")]] = parameters[[paste("b",
                                                              l, sep = "")]] - learning_rate * gradients[[paste("db",
                                                                                                                l, sep = "")]]
  }
  return(parameters)
}

predict_model <- function(parameters, X, hidden_layer_act, output_layer_act){
  pred <- numeric()
  scores <- forward_prop(X,
                         parameters,
                         hidden_layer_act,
                         output_layer_act)[['AL']]
  if(output_layer_act == 'softmax') {
    pred <- apply(scores, 1, which.max)
  }
  else{
    for(i in 1:ncol(scores)){
      if(scores[i] > 0.5) pred[i] = 1
      else pred[i] = 0
    }
  }
  return (pred)
}


n_layer_model <- function(X,
                          Y,
                          X_test,
                          Y_test,
                          layers_dims,
                          hidden_layer_act,
                          output_layer_act,
                          learning_rate,
                          num_iter,
                          initialization,
                          print_cost = F){
  set.seed(1)
  costs <- NULL
  parameters <- initialize_params(layers_dims, initialization)
  num_classes <- length(unique(Y))
  start_time <- Sys.time()
  for( i in 0:num_iter){
    AL = forward_prop(X,
                      parameters,
                      hidden_layer_act,
                      output_layer_act)[['AL']]
    caches = forward_prop(X,
                          parameters,
                          hidden_layer_act,
                          output_layer_act)[['caches']]
    cost <- compute_cost(AL,
                         X,
                         Y,
                         num_classes,
                         output_layer_act)
    gradients = back_prop(AL,
                          Y,
                          caches,
                          hidden_layer_act,
                          output_layer_act,
                          num_classes)
    parameters = update_params(parameters,
                               gradients,
                               learning_rate)
    costs <- c(costs, cost)
    if(print_cost == T & i %% 1000 == 0){
      cat(sprintf("Cost after iteration %d = %05f\n", i, cost))
    }
  }
  if(output_layer_act != 'softmax'){
    pred_train <- predict_model(parameters,
                                X,
                                hidden_layer_act,
                                output_layer_act)
    Tr_acc <- mean(pred_train == Y) * 100
    pred_test <- predict_model(parameters,
                               X_test,
                               hidden_layer_act,
                               output_layer_act)
    Ts_acc <- mean(pred_test == Y_test) * 100
    cat(sprintf("Cost after iteration %d, = %05f;
Train Acc: %#.3f, Test Acc: %#.3f, \n",
                i, cost, Tr_acc, Ts_acc))
  }
  else if(output_layer_act == 'softmax'){
    pred_train <- predict_model(parameters,
                                X,
                                hidden_layer_act,
                                output_layer_act)
    Tr_acc <- mean((pred_train - 1) == Y)
    pred_test <- predict_model(parameters,
                               X_test,
                               hidden_layer_act,
                               output_layer_act)
    Ts_acc <- mean((pred_test - 1) == Y_test)
    cat(sprintf("Cost after iteration , %d, = %05f;
Train Acc: %#.3f, Test Acc: %#.3f, \n",
                i, cost, Tr_acc, Ts_acc))
  }
  end_time <- Sys.time()
  cat(sprintf("Application running time: %#.3f minutes",
              (end_time - start_time) / 60 ))
  return(list("parameters" = parameters, "costs" = costs))
}













