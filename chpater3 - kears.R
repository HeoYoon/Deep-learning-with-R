library(keras)
library(EBImage)
library(pbapply)
library(tidyverse)
library(MASS)

file_path_train <- "C:/Users/Healthcare/Downloads/train_data"
file_path_test <- "C:/Users/Healthcare/Downloads/test_data"

height = 64
width = 64
channels = 3

extract_feature <- function(dir_path, width, height) {
  img_size <- width * height
  images <- list.files(dir_path)
  label <- ifelse(grepl("dog", images) == T, 1, 0)
  print(paste("Processing", length(images), "images"))
  feature_list <- pblapply(images, function(imgname) {
    img <- readImage(file.path(dir_path, imgname))
    img_resized <- EBImage::resize(img, w = width, h = height)
    img_matrix <- matrix(reticulate::array_reshape(img_resized, (width *
                                                                   height * channels)), nrow = width * height * channels)
    img_vector <- as.vector(t(img_matrix))
    return(img_vector)
  })
  feature_matrix <- do.call(rbind, feature_list)
  return(list(t(feature_matrix), label))
}
# Reshape the images
data_train <-extract_feature(file_path_train,width = 64,height = 64)
trainx <-data_train[[1]]
trainy <-data_train[[2]]
dim(trainx)

data_test <-extract_feature(file_path_test,width = 64,height = 64)
testx <-data_test[[1]]
testy<- data_test[[2]]
dim(testx)


set.seed(1)

model <- keras_model_sequential()

model %>%
  layer_dense(units = 100, activation = 'relu', input_shape = c(12288)) %>%
  layer_dense(units = 50, activation = 'relu') %>%
  layer_dense(units = 15, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'sigmoid') %>%
  compile(
    optimizer = optimizer_adam(),
    loss = 'binary_crossentropy',
    metrics = c('accuracy')
  )
summary(model)

hstory <- model %>%
  fit(t(trainx), trainy, epochs = 50, batch_size = 32,
      validation_split = 0.2)

history <- model %>%
  evaluate(t(trainx), trainy)
  
keras_pred_train <- model %>%
  predict_classes(t(trainx))


table(Predicted = keras_pred_train, Actual = trainy)






