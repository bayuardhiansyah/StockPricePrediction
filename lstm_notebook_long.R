---
title: "Forecasting with LSTM"
output: html_notebook
---

### Load library

```{r}
pacman::p_load(dplyr, ggplot2, lubridate, ggthemes, remotes,
               neuralnet, Metrics, tidyverse, readxl, knitr, 
               glue, forcats, timetk, tidyquant, tibbletime,
               recipes, rsample, yardstick, tictoc, kableExtra)

# install_github("rstudio/tensorflow") <- jalankan ini untuk pertama kali
library(tensorflow)
# install_tensorflow(method = "conda", envname = "r-reticulate") <- jalankan ini untuk pertama kali
library(keras)

```


### Load data


```{r}
mydata <- read_csv("UNVR.JK_long.csv", 
                   col_types = cols(Date = col_date(format = "%m/%d/%Y")))

# agar tidak terlalu berat dalam pengolahan, tidak seluruh data dilibatkan
# misal diambil 3000-an data dari Juni 2009

mydata <- mydata[1443:4642,]

```


### Visualisasi data time series

```{r}
mydata %>% 
  ggplot(aes(x = Date, y = Close)) +
  geom_line(color = "mediumpurple2") + 
  geom_point(size = 0.2, color = "slateblue3") +
  theme_hc() +
  labs(
    title = 'UNVR Price',
    subtitle = 'Period Jun-2009 to Apr-2022') +
  xlab("Date") + ylab("Close Price")

```


### Fokus pada kolom 'Close'

Untuk analisis selanjutnya, selain kolom 'Close' kolom lain diabaikan. Kemudian data dilakukan diferensiasi untuk lebih mendapatkan kestatsioneritasan.

```{r}
data <- mydata[,c(1,5)] %>% 
  pull(Close)

diffed <- diff(data, differences = 1)
head(diffed)

```

Setelah didiferensiasi, dibentuk vector dengan lag hasil diferensiasi. Tujuannya untuk menganalogikan adanya variabel input (X) dan variabel output (Y), karena pada LSTM mensyaratkan hal tersebut. Dalam hal ini (x-t) adalah input sedangkan (x) sebagai output.

```{r}

lag_transform <- function(x, k = 1){
  
  lagged =  c(rep(NA, k), x[1:(length(x)-k)])
  DF = as.data.frame(cbind(lagged, x))
  colnames(DF) <- c( paste0('x-', k), 'x')
  DF[is.na(DF)] <- 0
  return(DF)
}

supervised <- lag_transform(diffed, 1)
head(supervised)

```


### Splitting data; Training dan Testing

Alokasi 80:20

```{r}
N <- nrow(supervised)
n <- round(N * 0.8, digits = 0)
train <- supervised[1:n, ]
test  <- supervised[(n+1):N,  ]

```


### Normalisasi data


```{r}
scale_data <- function(train, test, feature_range = c(0, 1)) {
  x = train
  fr_min = feature_range[1]
  fr_max = feature_range[2]
  std_train = ((x - min(x) ) / (max(x) - min(x)  ))
  std_test  = ((test - min(x) ) / (max(x) - min(x)  ))
  
  scaled_train = std_train *(fr_max -fr_min) + fr_min
  scaled_test = std_test *(fr_max -fr_min) + fr_min
  
  return( list(scaled_train = as.vector(scaled_train), 
               scaled_test = as.vector(scaled_test),
               scaler= c(min =min(x), max = max(x))) )
  
}

```


```{r}
Scaled <- scale_data(train, test, c(-1, 1))

y_train <- Scaled$scaled_train[, 2]
x_train <- Scaled$scaled_train[, 1]

y_test <- Scaled$scaled_test[, 2]
x_test <- Scaled$scaled_test[, 1]

```


### Inverse transform

Fungsi ini digunakan untuk mengembalikan nilai yang sebelumnya dinormalisasi

```{r}
invert_scaling <- function(scaled, scaler, feature_range = c(0, 1)){
  min = scaler[1]
  max = scaler[2]
  t = length(scaled)
  mins = feature_range[1]
  maxs = feature_range[2]
  inverted_dfs = numeric(t)
  
  for( i in 1:t){
    X = (scaled[i]- mins)/(maxs - mins)
    rawValues = X *(max - min) + min
    inverted_dfs[i] <- rawValues
  }
  return(inverted_dfs)
}

```


### Modelling

Dalam LSTM diperlukan 3 dimensi array sebagain input shape.

```{r}
dim(x_train) <- c(length(x_train), 1, 1)
X_shape2 <- dim(x_train)[2]
X_shape3 <- dim(x_train)[3]
batch_size <- 1            
units <- 1 

```


Pembentukan model LSTM

```{r}
model <- keras_model_sequential() 
model %>% 
  layer_lstm(units, batch_input_shape = c(batch_size, X_shape2, X_shape3),
             stateful= TRUE) %>%
  layer_dense(units = 1)

```


Mengevaluasi model dengan parameter MSE

```{r}
model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam(learning_rate = 0.02, decay = 1e-6 ))

```


```{r}
summary(model)

```

### Hasil (progres) Evaluasi Model

```{r}
Epochs = 50   
for(i in 1:Epochs ){
  model %>% fit(x_train, y_train, epochs = 1, 
                 batch_size = batch_size, verbose = 1, shuffle = FALSE)
  model %>% reset_states()
}

```

## Membuat Prediksi

```{r}
L <- length(x_test)
scaler <- Scaled$scaler
prediksi <- numeric(L)

```


```{r}
for(i in 1:L){
  X = x_test[i]
  dim(X) = c(1,1,1)
  yhat = model %>% predict(X, batch_size = batch_size)
  yhat = invert_scaling(yhat, scaler,  c(-1, 1))
  yhat  = yhat + data[(n+i)]
  prediksi[i] <- yhat
}

```


### Menggabungkan Data

Penggabungan data aktual, training set, testing set, dan prediksi

```{r}
dataplot <- mydata[,c(1,5)]
train_plot <- dataplot[1:n, ]
test_plot  <- dataplot[(n+1):N,  ]

plotpred <- dataplot %>%
  rename(Actual = Close) %>%
  left_join(
    tibble(
      Date = train_plot$Date,
      Train = pull(train_plot, Close))) %>%
  left_join(
    tibble(
      Date = test_plot$Date,
      Test = pull(test_plot, Close))) %>%
  left_join(
    tibble(
      Date = test_plot$Date,
      Pred = prediksi))

```

### Membuat Plot

Plot yang menggambarkan data aktual (training + testing), training, testing, dan prediksi dari awal

```{r}
options(warn = -4)
plotpred %>% 
  gather(
    key = key, value = value,
    Actual, Train, Test, Pred
  ) %>%
  mutate(
    key = key %>% factor(levels = c("Actual", "Train", "Test", "Pred"))
  ) %>%
  ggplot(aes(x = Date, y = value, colour = key)) +
  geom_line(size = 1) + 
  scale_color_manual(values = c(
    "Actual" = "mediumpurple2",
    "Train" = "mediumpurple3",
    "Test" = "tomato2",
    "Pred" = "olivedrab2"
  )) + theme_hc() + 
  labs(
    title = 'Close Price',
    subtitle = 'Comparison between Actual and Predicted') +
  xlab("Date") + ylab("Price")

```

### Evaluasi Parameter Prediksi

```{r}
# fokus pada periode data testing
evaldata <- plotpred[2560:3199,]

rmse(evaldata$Test, evaldata$Pred)
mse(evaldata$Test, evaldata$Pred)

```

Untuk lebih memberikan gambaran jelas nilai antara testing dan prediksi dibentuk plot berdasarkan periode data testing

```{r}
evaldata %>% 
  gather(
    key = key, value = value,
    Test, Pred
  ) %>%
  mutate(
    key = key %>% factor(levels = c("Test", "Pred"))
  ) %>%
  ggplot(aes(x = Date, y = value, colour = key)) +
  geom_line(size = 0.6) + 
  scale_color_manual(values = c(
    "Test" = "tomato2",
    "Pred" = "olivedrab2"
  )) + theme_hc() + 
  labs(
    title = 'Close Price Test Dataset',
    subtitle = 'Comparison between Actual and Predicted') +
  xlab("Date") + ylab("Price")

```

Dari plot tersebut dan evaluasi parameter prediksi nampaknya nilai yang diprediksi pada data testing sudah cukup baik. 


### Forecasting

Misal dilakukan prediksi pengamatan untuk 120 periode ke depan, dibuatkan fungsi prediksinya sebagai berikut

```{r}
future_predict <- function(data, epochs = 300, ...) {
    
    lstm_prediction <- function(data, epochs, ...) {
        df <- data
        
        rec_obj <- recipe(value ~ ., df) %>%
            step_sqrt(value) %>%
            step_center(value) %>%
            step_scale(value) %>%
            prep()
        
        df_processed_tbl <- bake(rec_obj, df)
        
        center_history <- rec_obj$steps[[2]]$means["value"]
        scale_history  <- rec_obj$steps[[3]]$sds["value"]
        
        lag_setting  <- 120 #prediksi
        batch_size   <- 40
        train_length <- 440
        tsteps       <- 1
        epochs       <- epochs
        
        lag_train_tbl <- df_processed_tbl %>%
            mutate(value_lag = lag(value, n = lag_setting)) %>%
            filter(!is.na(value_lag)) %>%
            tail(train_length)
        
        x_train_vec <- lag_train_tbl$value_lag
        x_train_arr <- array(data = x_train_vec, dim = c(length(x_train_vec), 1, 1))
        
        y_train_vec <- lag_train_tbl$value
        y_train_arr <- array(data = y_train_vec, dim = c(length(y_train_vec), 1))
        
        x_test_vec <- y_train_vec %>% tail(lag_setting)
        x_test_arr <- array(data = x_test_vec, dim = c(length(x_test_vec), 1, 1))
                
        model <- keras_model_sequential()

        model %>%
            layer_lstm(units            = 50, 
                       input_shape      = c(tsteps, 1), 
                       batch_size       = batch_size,
                       return_sequences = TRUE, 
                       stateful         = TRUE) %>% 
            layer_lstm(units            = 50, 
                       return_sequences = FALSE, 
                       stateful         = TRUE) %>% 
            layer_dense(units = 1)
        
        model %>% 
            compile(loss = 'mean_squared_error', optimizer = 'adam')
        

        for (i in 1:epochs) {
            model %>% fit(x          = x_train_arr, 
                          y          = y_train_arr, 
                          batch_size = batch_size,
                          epochs     = 1, 
                          verbose    = 1, 
                          shuffle    = FALSE)
            
            model %>% reset_states()
            cat("Epoch: ", i)
            
        }
        
        # Membuat prediksi
        pred_out <- model %>% 
            predict(x_test_arr, batch_size = batch_size) %>%
            .[,1] 
        
        idx <- data %>%
            tk_index() %>%
            tk_make_future_timeseries(length_out = lag_setting)
        
        pred_tbl <- tibble(
            index   = idx,
            value   = (pred_out * scale_history + center_history)^2
            )
        
        tbl_1 <- df %>%
            add_column(key = "actual")

        tbl_3 <- pred_tbl %>%
            add_column(key = "predict")

        time_bind_rows <- function(data_1, data_2, index) {
            index_expr <- enquo(index)
            bind_rows(data_1, data_2) %>%
                as_tbl_time(index = !! index_expr)
        }

        ret <- list(tbl_1, tbl_3) %>%
            reduce(time_bind_rows, index = index) %>%
            arrange(key, index) %>%
            mutate(key = as_factor(key))

        return(ret)
        
    }
    
    safe_lstm <- possibly(lstm_prediction, otherwise = NA)
    
    safe_lstm(data, epochs, ...)
    
}

```


Hasil forecasting

```{r}
#membuat dataframe baru
data_for_predict <- mydata[,c(1,5)]

colnames(data_for_predict)[1] <- "index"
colnames(data_for_predict)[2] <- "value"

```


```{r}
#forecasting
tic()

future_value <- future_predict(data_for_predict, epochs = 300)

toc()

```

### Data aktual, prediksi, dan gabungan

```{r}

actual <- future_value[1:3200,]
prediksi <- future_value[3201:3320,]
prediksi$value <- prediksi$value/2
hasil_merge <- rbind(actual, prediksi)

```


Membuat tabel data prediksi

```{r}
prediksi %>%
  kbl() %>%
  kable_material(c("striped", "hover"))

```


Plot khusus data hasil prediksi

```{r}

prediksi %>% 
  ggplot(aes(x = index, y = value)) +
  geom_line(color = "olivedrab2", size = 1.2) + 
  geom_point(size = 0.5, color = "mintcream") +
  theme_solarized_2(light = FALSE) +
  labs(
    title = 'UNVR Forecasted Price',
    subtitle = '120 future point - Apr to Aug 2022') +
  xlab("Date") + ylab("Forecasted Close Price")

```


### Gabungan data prediksi dengan data awal

```{r}

hasil_fix <- hasil_merge %>%
  rename(Actual = value) %>%
  left_join(
    tibble(
      index = actual$index,
      Actual = pull(actual, value))) %>%
  left_join(
    tibble(
      index = prediksi$index,
      Prediksi = pull(prediksi, value))) 

```


Plot seluruhnya

```{r}
options(warn = -4)
hasil_fix %>% 
  gather(
    key = key, value = value,
    Actual, Prediksi
  ) %>%
  mutate(
    key = key %>% factor(levels = c("Actual", "Prediksi"))
  ) %>%
  ggplot(aes(x = index, y = value, colour = key)) +
  geom_line(size = 1) + 
  scale_color_manual(values = c(
    "Actual" = "mintcream",
    "Prediksi" = "olivedrab2"
  )) + theme_solarized_2(light = FALSE) + 
  labs(
    title = 'UNVR Price',
    subtitle = 'Actual (from 2021) and Future Prediction') +
  xlab("Date") + ylab("Price")

```

Untuk visualisasi lebih jelas, misalkan kita plot data mulai 2020

```{r}

options(warn = -4)
hasil_fix[2875:3320,] %>% 
  gather(
    key = key, value = value,
    Actual, Prediksi
  ) %>%
  mutate(
    key = key %>% factor(levels = c("Actual", "Prediksi"))
  ) %>%
  ggplot(aes(x = index, y = value, colour = key)) +
  geom_line(size = 1) + 
  scale_color_manual(values = c(
    "Actual" = "mintcream",
    "Prediksi" = "olivedrab2"
  )) + theme_solarized_2(light = FALSE) + 
  labs(
    title = 'UNVR Price',
    subtitle = 'Actual (from 2021) and Future Prediction') +
  xlab("Date") + ylab("Close Price")

```

## END
