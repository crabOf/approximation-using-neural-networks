library(torch)

#Подготовка данных
generate_data <- function(n) {
  t <- seq(0, 10, length.out = n)
  alpha <- -0.5
  beta <- sqrt(3) / 2
  c1 <- 4 / sqrt(3)
  c2 <- 2
  
  # Генерация аналитического решения
  response <- c1 * exp(alpha * t) * sin(beta * t) + c2 * exp(alpha * t) * cos(beta * t)
  
  return(data.frame(time = t, response = response))
}

# Генерация данных
n <- 1000
data <- generate_data(n)

# Преобразование данных в тензоры
time_tensor <- torch_tensor(data$time)
response_tensor <- torch_tensor(data$response)

# Определение модели
Model <- nn_module(
  "Model",
  initialize = function() {
    self$alpha <- nn_parameter(torch_tensor(0.0))
    self$c1 <- nn_parameter(torch_tensor(0.0))
    self$c2 <- nn_parameter(torch_tensor(0.0))
    self$beta <- sqrt(3) / 2
  },
  forward = function(t) {
    self$c1 * exp(self$alpha * t) * sin(self$beta * t) +
      self$c2 * exp(self$alpha * t) * cos(self$beta * t)
  }
)

# Обучение модели
model <- Model()
optimizer <- optim_adam(model$parameters, lr = 0.01)
criterion <- nn_mse_loss()

# Обучение
epochs <- 150
for (epoch in 1:epochs) {
  optimizer$zero_grad()
  
  # Прямой проход
  pred <- model(time_tensor)
  
  # Вычисление потерь
  loss <- criterion(pred, response_tensor)
  
  # Обратное распространение
  loss$backward()
  optimizer$step()
  
  if (epoch %% 10 == 0) {
    cat(sprintf("Epoch [%d/%d], Loss: %.4f\n", epoch, epochs, loss$item()))
  }
}

# Сравнение результатов
# Получение обученных параметров
estimation <- list(
  alpha = model$alpha$item(),
  c1 = model$c1$item(),
  c2 = model$c2$item()
)

# Аналитическое решение
analytic_solution <- function(t, alpha, beta, c1, c2) {
  c1 * exp(alpha * t) * sin(beta * t) + c2 * exp(alpha * t) * cos(beta * t)
}

# Построение графиков
library(ggplot2)

# Генерация данных для графика
t_values <- seq(0, 10, length.out = 100)
analytic_values <- analytic_solution(t_values, -0.5, sqrt(3) / 2, 4 / sqrt(3), 2)
approx_values <- analytic_solution(t_values, estimation$alpha, sqrt(3) / 2, estimation$c1, estimation$c2)

# Создание датафрейма для графика
plot_data <- data.frame(
  time = rep(t_values, 3),
  response = c(analytic_values, approx_values, rep(NA, length(t_values))),
  type = rep(c("Analytic", "Approximation", "Data"), each = length(t_values))
)

# Добавление данных
plot_data <- rbind(plot_data, data.frame(time = data$time, response = data$response, type = "Data"))

# Построение графика
ggplot(plot_data, aes(x = time, y = response, color = type)) +
  geom_line() +
  labs(title = "Аппроксимация функции с использованием нейронной сети",
       x = "Время (t)",
       y = "Ответ (x )") +
  theme_minimal() +
  scale_color_manual(values = c("blue", "red", "green")) +
  theme(legend.title = element_blank())