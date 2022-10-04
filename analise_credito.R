#Definindo local de trabalho
#setwd("D:/Data_Science/Projetos/analise_credito_ML")

#carregando bibliotecas
#library(dplyr)
#library(smotefamily)
#devtools::install_github("dongyuanwu/RSBID")
library(RSBID)
library(randomForest)
library(ROCR)
library(caret)
library(pROC)
library(ggplot2)
library(gridExtra)
library(e1071)

#Carregando dataset
df <- read.csv("credito.csv")

#Explorando dados
head(df)
str(df)
any(is.na(df))
summary(df[,c(6,14,3)]) 
table(df$credit.rating)

#Histograma das variáveis "credit.amount" e "age"
g1<-ggplot(df, aes(x = credit.amount)) + 
  geom_histogram(bins = 20, 
                 alpha = 0.5, fill = 'blue') + 
  theme_minimal()

g2<-ggplot(df, aes(x = age)) + 
  geom_histogram(bins = 20, 
                 alpha = 0.5, fill = 'blue') + 
  theme_minimal()
grid.arrange(g1 , g2 , ncol=2)

#transformar em variáveis categóricas
for (i in colnames(df,do.NULL = FALSE,prefix = "col")){
  if(i!="credit.duration.months"&i!="credit.amount"&i!="age")
  df[,i] <- as.factor(df[,i])
}
str(df)

## Normalização
scale.features <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- scale(df[[variable]], center=T, scale=T)
  }
  return(df)
}

# Normalizando/padronizando as variáveis numéricas
numeric.vars <- c("credit.duration.months", "age", "credit.amount")
df <- scale.features(df, numeric.vars)

#Split dos dados
indexes <- sample(1:nrow(df), size = 0.7 * nrow(df))
df_train <- df[indexes,]
df_test <- df[-indexes,]
str(df_train)
str(df_test)

#Transformação (necessária para o balanceamento)
df_train$credit.duration.months<-as.double(df_train$credit.duration.months)
df_train$credit.amount<-as.double(df_train$credit.amount)
df_train$age<-as.double(df_train$age)

df_test$credit.duration.months<-as.double(df_test$credit.duration.months)
df_test$credit.amount<-as.double(df_test$credit.amount)
df_test$age<-as.double(df_test$age)

#Balanceando dados de treino
ggplot(df_train, aes(x = credit.rating)) + geom_bar(alpha = 0.5, fill = 'blue') + theme_minimal()
table(df_train$credit.rating)

df_train <- SMOTE_NC(df_train, 'credit.rating', perc_maj = 100, k = 5)

table(df_train$credit.rating)
sum(duplicated(df_train))
str(df_train)
ggplot(df_train, aes(x = credit.rating)) + geom_bar(alpha = 0.5, fill = 'blue') + theme_minimal()

#CONSTRUINDO MODELO COM RANDOM FOREST
modelo_RF <- randomForest( credit.rating ~ .,
                        data = df_train, 
                        ntree = 100, nodesize = 10, importance = T)
varImpPlot(modelo_RF)

modelo_RF <- randomForest( credit.rating ~ . -foreign.worker
                        -dependents
                        -other.credits
                        -occupation
                        -bank.credits
                        -apartment.type
                        -telephone
                        -marital.status
                        -guarantor,
                        data = df_train, 
                        ntree = 100, nodesize = 10, importance = T)
varImpPlot(modelo_RF)
print(modelo_RF)

# Gerando Confusion Matrix com o Caret
confusionMatrix(df_train$credit.rating, modelo_RF$predicted, positive = '1')

# Gerando previsoes nos dados de teste
result_previsto_RF <- data.frame( atual = df_test$credit.rating,
                               previsto = predict(modelo_RF, newdata = df_test))

# Gerando Confusion Matrix com o Caret
confusionMatrix(result_previsto_RF$atual, result_previsto_RF$previsto, positive = '1')

# Gerando as classes de dados
class1_RF <- predict(modelo_RF, newdata = df_test, type = 'prob')
class2_RF <- df_test$credit.rating

# Gerando a curva ROC e valor da AUC
pred_RF <- prediction(class1_RF[,2], class2_RF)
perf_RF <- performance(pred_RF, "tpr","fpr") 
auc_RF <- performance(pred_RF, "auc")
auc_RF <- auc_RF@y.values
auc_RF <- auc_RF[[1]]
auc_RF <- round(auc_RF,3)
plot(perf_RF, col = rainbow(10),main=paste("Curva ROC","\n","AUC=",auc_RF))


#CONSTRUINDO MODELO COM REGRESSÃO LOGÍSTICA

modelo_RL <- glm(credit.rating ~ . -foreign.worker
                -dependents
                -other.credits
                -occupation
                -bank.credits
                -apartment.type
                -telephone
                -marital.status
                -guarantor, 
                data = df_train, family = "binomial")

# Visualizando o modelo
summary(modelo_RL)

# Gerando Confusion Matrix com o Caret
confusionMatrix(df_train$credit.rating, as.factor(round(modelo_RL$fitted.values)), positive = '1')

# Gerando previsoes nos dados de teste
result_previsto_RL <- data.frame( atual = df_test$credit.rating,
                                  previsto = as.factor(round(predict(modelo_RL, newdata = df_test, type = 'response'))))

# Gerando Confusion Matrix com o Caret
confusionMatrix(result_previsto_RL$atual, result_previsto_RL$previsto, positive = '1')

# Gerando as classes de dados
class1_RL <- predict(modelo_RL, newdata = df_test, type = 'response')
class2_RL <- df_test$credit.rating

# Gerando a curva ROC
pred_RL <- prediction(class1_RL, class2_RL)
perf_RL <- performance(pred_RL, "tpr","fpr") 
auc_RL <- performance(pred_RL, "auc")
auc_RL <- auc_RL@y.values
auc_RL <- auc_RL[[1]]
auc_RL <- round(auc_RL,3)
plot(perf_RL,col = rainbow(10),main=paste("Curva ROC","\n","AUC=",auc_RL))

#CONSTRUINDO MODELO COM NAIVE BAYES
modelo_NB <- naiveBayes( credit.rating ~ . -foreign.worker
                         -dependents
                         -other.credits
                         -occupation
                         -bank.credits
                         -apartment.type
                         -telephone
                         -marital.status
                         -guarantor,
                         data = df_train)

print(modelo_NB)

# Gerando previsoes nos dados de teste
result_previsto_NB <- data.frame( atual = df_test$credit.rating,
                                  previsto = predict(modelo_NB, df_test[,-1]))

# Gerando Confusion Matrix com o Caret
confusionMatrix(result_previsto_NB$atual, result_previsto_NB$previsto, positive = '1')

# Gerando as classes de dados
class1_NB <- predict(modelo_NB, newdata = df_test[,-1],type = c("raw"))
class2_NB <- df_test$credit.rating

# Gerando a curva ROC e valor da AUC
pred_NB <- prediction(class1_NB[,2], class2_NB)
perf_NB <- performance(pred_NB, "tpr","fpr") 
auc_NB <- performance(pred_NB, "auc")
auc_NB <- auc_NB@y.values
auc_NB <- auc_NB[[1]]
auc_NB <- round(auc_NB,3)
plot(perf_NB, col = rainbow(10),main=paste("Curva ROC","\n","AUC=",auc_NB))

#Comparando modelos (biblioteca pROC)
roc_RL <- roc(class2_RL , class1_RL, percent = TRUE)
roc_RF <- roc(class2_RF , class1_RF[,2], percent = TRUE)
roc_NB <- roc(class2_NB , class1_NB[,2], percent = TRUE)
par(pty = "s")
plot(roc_RL, print.auc = TRUE, col = "blue", main=paste("Random F. (Green) x Regressão Log. (Blue) x Naive Bayes (Orange)"), legacy.axes = TRUE, 
     xlab = "% de Falso Positivo (100 - Especificidade)",
     ylab = "% de Verdadeiro Positivo (Sensibilidade)")
plot(roc_RF, print.auc = TRUE, col = "green", print.auc.y = 40, add = TRUE, legacy.axes = TRUE)
plot(roc_NB, print.auc = TRUE, col = "orange", print.auc.y = 30, add = TRUE, legacy.axes = TRUE)
#dev.off()


