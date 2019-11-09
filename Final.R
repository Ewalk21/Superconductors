setwd("D:\\Mathemagic\\Fall2019\\CS614_Interactive_Data_Analysis\\Final_Project")
rm(list=ls())
#install.packages("sqldf")
library(ggplot2)
library(sqldf)
#install.packages("xgboost")
library(xgboost)
library(CHNOSZ)
library(dplyr)
#install.packages("ranger")
library(ranger)
#tmp_rf_model = ranger(critical_temp ~ ., data = train, num.trees = 2500)

#There are two files: (1) train.csv contains 81 features extracted from 21263 superconductors 
#along with the critical temperature in the 82nd column, (2) unique_m.csv contains the chemical 
#formula broken up for all the 21263 superconductors from the train.csv file. 
#The last two columns have the critical temperature and chemical formula
#https://www.nature.com/articles/s41524-018-0085-8
#https://www.sciencedirect.com/science/article/pii/S0927025618304877?via%3Dihub
#https://github.com/khamidieh/predict_tc
#Features extracted based on thermal conductivity, atomic radius, valence, electron affinity,
#and atomic mass contribute the most to the model's predictive accuracy

df=read.csv("train.csv")
uniq=read.csv("unique_m.csv")
summary(df)
summary(uniq)
names(df)
str(df)
str(uniq)
plot(df$critical_temp,df$mean_atomic_mass)
hist(df$wtd_mean_atomic_mass)
plot(df$mean_atomic_mass)

plot(df$wtd_mean_atomic_mass,df$mean_atomic_mass)
cor(df$wtd_mean_atomic_mass,df$mean_atomic_mass)
plot(df$wtd_mean_atomic_mass,df$wtd_gmean_atomic_mass)
cor(df$wtd_mean_atomic_mass,df$wtd_gmean_atomic_mass)

hist(df$mean_Valence)
hist(df$wtd_mean_Valence)
plot(df$mean_Valence)
plot(df$critical_temp,df$wtd_mean_Valence)
hist(df$critical_temp)
hist(uniq$Ba)
hist(uniq$Cu)
hist(uniq$O)
hist(uniq$Fe)
hist(uniq$H)
cor(df$gmean_atomic_mass,df$wtd_gmean_atomic_mass)
cor(df$wtd_mean_atomic_mass,df$wtd_gmean_atomic_mass)
cor(uniq$O,uniq$critical_temp)
cor(uniq$Cu,uniq$critical_temp)
summary(df$critical_temp)

summary(lm(critical_temp~ number_of_elements, data=df))
summary(lm(critical_temp~ std_atomic_mass, data=df))
summary(lm(critical_temp~ wtd_mean_ThermalConductivity, data=df))
summary(lm(critical_temp~ Be, data=uniq))
summary(lm(critical_temp~ range_ThermalConductivity , data=df))
plot(df$critical_temp,df$range_ThermalConductivity)

plot(df$critical_temp,df$std_atomic_mass)
hg_meancrit=uniq[which(uniq$Hg > 0 ),]$critical_temp
hist(hg_meancrit,breaks=30)
summary(hg_meancrit)
v=sd(hg_meancrit)

c=t(rep(0,86))
mean_crit = as.data.frame(c)
names(mean_crit)=names(uniq[,-c(88,87)])

size=dim(uniq_test)
mean_crit_all = c()
uniq_test=uniq[,-c(88)]
names(mean_crit)=names(uniq[,-c(88,87)])
size=dim(uniq_test)
for (ii in 1:size[2]-1) {
  nonZero=uniq_test[which(uniq_test[,ii] > 0),]
  crit_tempii=nonZero$critical_temp
  meancrit=mean(crit_tempii)
  mean_crit_all=rbind(mean_crit_all,meancrit)
}
mean_crit_all
#mean_crit_all[,0]=names(uniq_test)
plot(sort(mean_crit_all[,1]))

c=t(rep(0,86))
mean_crit = as.data.frame(c)
names(mean_crit)=names(uniq[,-c(88,87)])
size=dim(uniq_test)
ii=1

for (name in names(uniq[,-c(88,87)])) {
  elem=paste(name,"1",sep="")
  element=uniq[which(uniq$material==elem),]
  mean_crit_element=mean(element$critical_temp)
  if (is.nan(mean_crit_element)){
    elem=paste(name,"2",sep="")
    element=uniq[which(uniq$material==elem),]
    mean_crit_element=mean(element$critical_temp)
  }
  mean_crit[1,ii]=mean_crit_element
  ii = ii+1
}
#mean_crit[0,]=names(uniq[,-c(88,87)])
mean_crit=sort(mean_crit)
dim(mean_crit)
plot(mean_crit[1,])

m=uniq[which(uniq$material=="C1"),]
View(m)
plot(uniq[which(uniq$material=="C1"),]$critical_temp)


#Models:
#PCA
#Regression (Baseline)
#Random Forest, or tree regression
# eXtreme Gradient Boosting package.
#multilayer perceptron?




#PCA example
#https://rstudio-pubs-static.s3.amazonaws.com/92006_344e916f251146daa0dc49fef94e2104.html

##______differences between Uniq and df______________
pca_object = princomp(x = uniq[,-c(87,88)])
summary(pca_object)
plot(pca_object)
pca_object$loadings
screeplot(pca_object)

cum_var = cumsum(pca_object$sdev^2)/sum(pca_object$sdev^2)
plot(cum_var, type = "b", pch = 19, xlab = "Number of Components", 
     ylab = "Cummulative Variance")
abline(h = c(0.9,0.99), lty = 2)

baseline=lm(critical_temp~. , data=df)
baseline2=lm(critical_temp~. , data=uniq[,-c(88)])
anova(baseline,baseline2)

sigidx=c(28,65,77,7,73,45,49,63,75,10,40,67,69,3,11,34,82) #71,68,70,51
df_new=df[,sigidx]
baseline=lm(critical_temp~. , data=df_new)
summary(baseline)


scale_it = function(x)
{
  out = (x - mean(x))/sd(x)
  out
}
scaled_df = df

for (i in 1:81)
{
  scaled_df[,i] = scale_it(scaled_df[,i])
}

sigidx=c(28,65,77,7,73,45,49,63,75,10,40,67,69,3,11,34,82) #71,68,70,51
df_new=scaled_df[,sigidx]
baseline=lm(critical_temp~. , data=df_new)
summary(baseline)

train.idx <- sample(nrow(df), 0.8 * nrow(df))
df_train <- df[train.idx, ]
df_test <- df[-train.idx, ]

train.idx <- sample(nrow(df_new), 0.8 * nrow(df_new))
df_new_train <- df_new[train.idx, ]
df_new_test <- df_new[-train.idx, ]


tmp_rf_model = ranger(critical_temp ~ ., data = df_train, num.trees = 2500)
summary(tmp_rf_model)
pred=predict(tmp_rf_model,data=df_test)
RMSE=sqrt(sum((pred$predictions-train_test$critical_temp)^2)/dim(train_test)[1])
#9.667

tmp_rf_model2 = ranger(critical_temp ~ ., data = train_new_train, num.trees = 5000)
pred2=predict(tmp_rf_model2,data=train_new_test)
RMSE2=sqrt(sum((pred2$predictions-train_new_test$critical_temp)^2)/dim(train_new_test)[1])
RMSE2
#9.50712

install.packages('gbm')
library(gbm)
gdmod = gbm(critical_temp ~ ., data = df_new_train, distribution = "gaussian",
                  interaction.depth = 14, shrinkage = 0.10, n.trees = 2500)
best.iter <- gbm.perf(gdmod, method = "OOB")
Yhat <- predict(gdmod, newdata = df_new_test, n.trees = best.iter, type = "link")
# least squares error
print(sqrt(sum((df_new_test$critical_temp - Yhat)^2)/dim(df_new_test)[1]))


logreg=glm(critical_temp ~ ., data = df_new_train,family=gaussian)
summary(logreg)
anova(logreg)
pred3=predict(logreg,newdata=df_new_test,type="response")
RMSE3=sqrt(sum((pred3-df_new_test$critical_temp)^2)/dim(df_new_test)[1])
RMSE3 #19.555

baseline=lm(critical_temp~. , data=df_new_train)
pred4=predict(baseline,newdata=df_new_test,type="response")
RMSE4=sqrt(sum((pred3-df_new_test$critical_temp)^2)/dim(df_new_test)[1])
RMSE4
summary(baseline)
# range_ThermalConductivity        0.295
# wtd_std_ThermalConductivity      0.084
# range_atomic_radius              0.072
# wtd_gmean_ThermalConductivity    0.047
# std_ThermalConductivity          0.042
# wtd_entropy_Valence              0.038
# wtd_std_ElectronAffinity         0.036
# wtd_entropy_atomic_mass          0.025
# wtd_mean_Valence                 0.022
# wtd_gmean_ElectronAffinity       0.021
# wtd_range_ElectronAffinity       0.016
# wtd_mean_ThermalConductivity     0.015
# wtd_gmean_Valence                0.014
# std_atomic_mass                  0.013
# std_Density                      0.010
# wtd_entropy_ThermalConductivity  0.010
# wtd_range_ThermalConductivity    0.010
# wtd_mean_atomic_mass             0.009
# wtd_std_atomic_mass              0.009
# gmean_Density                    0.009

namestr=c("number_of_elements","mean_atomic_mass","wtd_mean_atomic_mass","gmean_atomic_mass",
            "wtd_gmean_atomic_mass","entropy_atomic_mass","wtd_entropy_atomic_mass",
            "range_atomic_mass","wtd_range_atomic_mass","wtd_std_atomic_mass","std_atomic_mass","mean_fie")
sigindex=which(colnames(df)==namestr)

#PCA loadings
# number_of_elements        
# mean_atomic_mass                 
# wtd_mean_atomic_mass                 
# gmean_atomic_mass                   
# wtd_gmean_atomic_mass             
# entropy_atomic_mass                                               
# wtd_entropy_atomic_mass                            
# range_atomic_mass           
# wtd_range_atomic_mass       
# std_atomic_mass               
# wtd_std_atomic_mass      
# mean_fie


#__________MAIN ANALYSIS_______________
dim(df)
set.seed(1234)
train.idx <- sample(nrow(df), 0.8 * nrow(df))
X_train <- df[train.idx, ]
X_test <- df[-train.idx, ]
y_train <- X_train[,82]
y_test <- X_test[,82]

X_pca = prcomp(X_train[,-82],center=T,scale.=T, rank.=30)
X_test_pca=as.data.frame(predict(X_pca,X_test[,-82]))
X_train_pca=as.data.frame(predict(X_pca,X_train[,-82]))

summary(X_test_pca)
summary(X_train_pca)
dim(X_test_pca)
dim(X_train_pca)
dim(df)
head(X_train_pca)
write.csv(y_train,'y_train.csv')
write.csv(y_test,'y_test.csv')
write.csv(X_train_pca,'X_train_pca.csv')
write.csv(X_test_pca,'X_test_pca.csv')


#___PCA ANALYSIS____
baseline1=lm(sqrt(X_train$critical_temp)~. , data=X_train_pca)
pred1=predict(baseline1,newdata=X_test_pca,type="response")
RMSE1=sqrt(sum(((pred1)^2-y_test)^2)/length(y_test))
RMSE1 #18.42
summary(baseline1) #R2 .73

baseline2=lm(log(X_train$critical_temp)~. , data=X_train_pca)
pred2=predict(baseline2,newdata=X_test_pca,type="response")
RMSE2=sqrt(sum((exp(pred2)-y_test)^2)/length(y_test))
RMSE2 #20.01
summary(baseline2) #R2 .67

baseline3=lm(X_train$critical_temp~. , data=X_train_pca)
pred3=predict(baseline3,newdata=X_test_pca,type="response")
RMSE3=sqrt(sum((pred3-y_test)^2)/length(y_test))
RMSE3 #19.13
summary(baseline3) #R2 .68

library(MASS)
summary(baseline1)
stepmod=stepAIC(baseline1, direction = "both",data=X_train_pca)
summary(stepmod) #R2 .73
spred1=predict(stepmod,newdata=X_test_pca,type="response")
sRMSE1=sqrt(sum(((spred1)^2-y_test)^2)/length(y_test))
sRMSE1 #18.43
#stepmod2=stepAIC(stepmod,~.^2, direction = "forward",data=X_train_pca)

library(gbm)
gdmod3 = gbm(sqrt(y_train) ~ ., data = X_train_pca, distribution = "gaussian",
            interaction.depth = 16, shrinkage = 0.20, n.trees = 2500)
best.iter <- gbm.perf(gdmod3, method = "OOB")
Yhat <- predict(gdmod3, newdata =X_test_pca, n.trees = best.iter, type = "link")
RMSE6=sqrt(sum((y_test - Yhat^2)^2)/length(y_test))
RMSE6 #12.90
summary(gdmod3) #PC1



#___RAW ANALYSIS_____
baseline1=lm(sqrt(X_train$critical_temp)~. , data=X_train)
pred1=predict(baseline1,newdata=X_test,type="response")
RMSE1=sqrt(sum(((pred1)^2-y_test)^2)/length(y_test))
RMSE1 #16.89
summary(baseline1) # R2 .77

baseline2=lm(log(X_train$critical_temp)~. , data=X_train)
pred2=predict(baseline2,newdata=X_test,type="response")
RMSE2=sqrt(sum((exp(pred2)-y_test)^2)/length(y_test))
RMSE2 #18.70
summary(baseline2) # R2 .71

baseline3=lm(X_train$critical_temp~. , data=X_train)
pred3=predict(baseline3,newdata=X_test,type="response")
RMSE3=sqrt(sum((pred3-y_test)^2)/length(y_test))
RMSE3 #17.55
summary(baseline3) # R2 .73

summary(baseline1)
stepmod=stepAIC(baseline1, direction = "both",data=X_train)
summary(stepmod) #R2 .77
spred1=predict(stepmod,newdata=X_test,type="response")
sRMSE1=sqrt(sum(((spred1)^2-y_test)^2)/length(y_test))
sRMSE1 #16.89

library(gbm)
gdmod = gbm(critical_temp ~ ., data = X_train, distribution = "gaussian",
            interaction.depth = 16, shrinkage = 0.20, n.trees = 2500)
best.iter <- gbm.perf(gdmod, method = "OOB")
Yhat <- predict(gdmod, newdata =X_test, n.trees = best.iter, type = "link")
RMSE5=sqrt(sum((y_test - Yhat)^2)/length(y_test))
RMSE5  #11.61
summary(gdmod)  #range thermal conuctivity super important

gdmod2 = gbm(sqrt(critical_temp) ~ ., data = X_train, distribution = "gaussian",
            interaction.depth = 14, shrinkage = 0.10, n.trees = 2500)
best.iter <- gbm.perf(gdmod2, method = "OOB")
Yhat <- predict(gdmod2, newdata =X_test, n.trees = best.iter, type = "link")
RMSE6=sqrt(sum((y_test - Yhat^2)^2)/length(y_test))
RMSE6 #11.81
summary(gdmod2)

#______________SCALED ANALSIS____________
scale_it = function(x)
{
  out = (x - mean(x))/sd(x)
  out
}
scaled_df = df

for (i in 1:81)
{
  scaled_df[,i] = scale_it(scaled_df[,i])
}
d_new = scaled_df
dim(df)
set.seed(1234)
train.idx <- sample(nrow(scaled_df), 0.8 * nrow(scaled_df))
Xs_train <- scaled_df[train.idx, ]
Xs_test <- scaled_df[-train.idx, ]
ys_train <- Xs_train[,82]
ys_test <- Xs_test[,82]

baseline1=lm(sqrt(Xs_train$critical_temp)~. , data=Xs_train)
pred1=predict(baseline1,newdata=Xs_test,type="response")
RMSE1=sqrt(sum(((pred1)^2-ys_test)^2)/length(ys_test))
RMSE1 #16.89
summary(baseline1) # R2 .77

baseline2=lm(log(Xs_train$critical_temp)~. , data=Xs_train)
pred2=predict(baseline2,newdata=Xs_test,type="response")
RMSE2=sqrt(sum((exp(pred2)-ys_test)^2)/length(ys_test))
RMSE2 #18.70
summary(baseline2) # R2 .71

baseline3=lm(Xs_train$critical_temp~. , data=Xs_train)
pred3=predict(baseline3,newdata=Xs_test,type="response")
RMSE3=sqrt(sum((pred3-ys_test)^2)/length(ys_test))
RMSE3 #17.55
summary(baseline3) # R2 .73

summary(baseline1)
stepmod=stepAIC(baseline1, direction = "backward",data=Xs_train)
summary(stepmod) #R2 .77
spred1=predict(stepmod,newdata=Xs_test,type="response")
sRMSE1=sqrt(sum(((spred1)^2-ys_test)^2)/length(ys_test))
sRMSE1 #16.89



#_______________RANDOM REGRESSION FOREST_____________
rf_model = ranger(y_train ~ ., data = X_train_pca, num.trees = 5000)
summary(rf_model)
pred=predict(rf_model,data=X_test_pca)
RMSE=sqrt(sum((pred$predictions-y_test)^2)/length(y_test))
RMSE
#pca: 9.36
rf_model = ranger(critical_temp ~ ., data = X_train, num.trees = 5000)
summary(rf_model)
pred=predict(rf_model,data=X_test)
RMSE=sqrt(sum((pred$predictions-y_test)^2)/length(y_test))
RMSE
#raw: 8.8

plot(X_train$critical_temp,X_train_pca$PC1)
plot(X_train$critical_temp,X_train_pca$PC2)
plot(X_train$critical_temp,X_train_pca$PC3)
plot(X_train$critical_temp,X_train_pca$PC4)
plot(X_train$critical_temp,X_train_pca$PC5)

#install.packages("PerformanceAnalytics")
library(PerformanceAnalytics)
chart.Correlation(X_train_pca)


#_____________Clustering_________Identify type1,type2
require(gridBase)
library(cluster)
library(e1071)
#install.packages("doParallel")
#library(doParallel)
#registerDoParallel(cores=8)
#getDoParWorkers()
train.idx <- sample(nrow(df), 0.4 * nrow(df))
X_train <- df[train.idx, ]
d2=dist(X_train[,1:80])
h1 = hclust(d2, method = "single")
h2 = hclust(d2, method = "complete")
h3 = hclust(d2, method = "average")
plot(h1)
plot(h2)
plot(h3)

K=3
cut_h2 = factor(cutree(h2, k = K))
#train$clstr = cut_h2

silstore = NA
for(ii in 2:10){
  kmn = kmeans(d2,ii)
  sil = silhouette(kmn$cluster, d2)
  silstore = c(silstore,mean(sil[,3]))
  print(ii)
}
plot(1:10,silstore, type = "o")
silstore

##### 7 - Gap Statistic #####
gap = clusGap(oil_nolab,FUN = kmeans,K.max = 8)
plot(gap)



d3=dist(df_pca)
silstore2 = NA
for(ii in 2:10){
  kmn = kmeans(d3,ii)
  sil = silhouette(kmn$cluster, d3)
  silstore2 = c(silstore2,mean(sil[,3]))
  print(ii)
}
plot(1:10,silstore2, type = "o")
silstore2





#________Uniq_PCA_Analysis______
set.seed(1234)
uniq_train.idx <- sample(nrow(uniq), 0.8 * nrow(uniq))
X_U_train <- uniq[uniq_train.idx, ]
X_U_test <- uniq[-uniq_train.idx, ]
y_U_train <- X_U_train[,88]
y_U_test <- X_U_test[,88]

X_Upca = prcomp(X_U_train[,-c(87,88)], rank.=30)
X_test_Upca=as.data.frame(predict(X_Upca,X_U_test[,-c(87,88)]))
X_train_Upca=as.data.frame(predict(X_Upca,X_U_train[,-c(87,88)]))
summary(X_test_Upca)
summary(X_train_Upca)
dim(X_test_Upca)
dim(X_train_Upca)
dim(uniq)


baseline=lm(X_U_train$critical_temp~. , data=X_train_Upca)
summary(baseline)
pred5=predict(baseline,newdata=X_test_Upca,type="response")
RMSE5=sqrt(sum((pred5-X_U_test$critical_temp)^2)/dim(X_test_Upca)[1])
RMSE5 #19.555

logreg=glm(X_U_train$critical_temp ~ ., data = X_train_Upca, family="gaussian")
summary(logreg)
anova(logreg)
pred3=predict(logreg,newdata=X_test_Upca,type="response")
RMSE3=sqrt(sum((pred3-X_U_test$critical_temp)^2)/dim(X_test_Upca)[1])
RMSE3 #27.52




#_______factorize critical temp analysis________
df_cut=df
quantile(df$critical_temp)
df_cut$critical_temp=cut(df$critical_temp,breaks=c(-Inf,0.5,5.3,20,63,100,Inf),
                                labels=c("verylow","low","medium","high","veryhigh","superhigh"))
names(df)
plot(df_cut$critical_temp)
dim(df)
set.seed(1234)
train.idx <- sample(nrow(df_cut), 0.8 * nrow(df_cut))
X_Ctrain <- df_cut[train.idx, ]
X_Ctest <- df_cut[-train.idx, ]
y_Ctrain <- X_Ctrain[,82]
y_Ctest <- X_Ctest[,82]
write.csv(y_Ctrain,'y_Ctrain.csv')
write.csv(y_Ctest,'y_Ctest.csv')
write.csv(X_Ctrain,'X_Ctrain.csv')
write.csv(X_Ctest,'X_Ctest.csv')


X_Cpca = prcomp(X_Ctrain[,-82],center=T,scale.=T, rank.=30)
X_Ctest_pca=as.data.frame(predict(X_Cpca,X_Ctest[,-82]))
X_Ctrain_pca=as.data.frame(predict(X_Cpca,X_Ctrain[,-82]))
X_Ctest_pca=cbind(X_Ctest_pca,X_Ctest$critical_temp)
X_Ctrain_pca=cbind(X_Ctrain_pca,X_Ctrain$critical_temp)
names(X_Ctrain_pca)[31]="critical_temp"
names(X_Ctest_pca)[31]="critical_temp"
summary(X_Ctest_pca)
summary(X_Ctrain_pca)
dim(X_Ctest_pca)
dim(X_Ctrain_pca)
dim(df_cut)

baseline=lm(critical_temp~. , data=X_Ctrain_pca)
summary(baseline)
pred6=predict(baseline,newdata=X_Ctest_pca,type="response")
#RMSE6=sqrt(sum((pred6-X_Ctest_pca$critical_temp)^2)/dim(X_Ctest_pca)[1])
#RMSE6 #19.555
auc(critical_temp~pred6,data=X_Ctest_pca,plot=T)
roc(baseline,plot=T,data=X_Ctest_pca)

logreg=glm(critical_temp ~ ., data = X_Ctrain_pca, family="binomial")
summary(logreg)
anova(logreg)
pred7=predict(logreg,newdata=X_Ctest_pca,type="response")
#correct=apply()
acc=sum(pred7==X_Ctest_pca$critical_temp)/dim(X_Ctest_pca)[1]
acc
auc(critical_temp~pred7,data=X_Ctest_pca,plot=T)
roc(logreg,plot=T,data=X_Ctest_pca)
roc(X_Ctest_pca$critical_temp~pred7,plot=T)

#install.packages("pROC")
library(pROC)
roc(critical_temp~pred7)
roc(logreg,plot=T,data=X_test_Cpca)

auc(critical_temp~po,data=X_test_Cpca,plot=T)


#po=predict(fit,type="response")


#_______________RANDOM classification FOREST_____________
rf_model = ranger(y_Ctrain ~ ., data = X_Ctrain, num.trees = 5000)
summary(rf_model)
pred=predict(rf_model,data=X_Ctest)
acc=sum(pred$predictions==y_Ctest)/length(y_Ctest)
acc
#acc: .98

install.packages('confusionMatrix')
confusionMatrix(pred$predictions,y_Ctest)








###____________________________________PLOTS__________________________________________
#_____PLOT_1____ critical temp distribution
#install.packages("fitdistr")
library(MASS)
summary(df$critical_temp)
m=fitdistr(df$critical_temp,"normal")
summary(df$critical_temp[df$critical_temp > 150])
g=ggplot(df, aes(x = critical_temp))+
  geom_histogram(aes(y=..density..),col="black", fill = "darkgreen", alpha = 0.4,bins=45)+
  geom_density(col = "red",fill = "blue", alpha = 0.2, size = .8)+
  ggtitle("Distribution of Critical Temperature of Superconductors")+
  ylab("Count")+
  scale_x_continuous(name ="Critical Temperature (K)",limits=c(0,140),breaks=seq(0,190,by=10))
g
ggsave("gg_crit_temp_hist.png", plot=g)



#_____PLOT_2____ PCA
#data=X_train_pca
#data$cum_var=cum_var
X_pca_plt = prcomp(df[,-82],center=T,scale.=T)
df_pca=as.data.frame(predict(X_pca_plt,df[,-82]))
cum_var = cumsum(X_pca_plt$sdev^2)/sum(X_pca_plt$sdev^2)
cor_var = as.vector(apply(df_pca, MARGIN = 2, function(z) cor(z,train$critical_temp,method="pearson")))
data_plt= as.data.frame(cbind((cor_var),(cum_var),1:81))
names(data_plt)=c("cor","cumvar","PrinComp")
#c=seq(10,31,by=3)
string=paste("variance explained:",round(1000*data_plt[30,2])/1000)

g1=ggplot(data_plt,aes(x=PrinComp,y=cumvar))+
  geom_line(col="darkblue")+
  geom_point(aes(alpha=cor),col="darkgreen",size=3.5)+
  labs(alpha="Correlation",caption="(Correlation coloring based on the correlation 
       of a principal component with critical temperature)")+
  ggtitle("Cumulative variance explained by each additional Prinicipal Component")+
  geom_hline(yintercept = .9, col = "red3",linetype = "dashed", size = .5)+
  geom_vline(xintercept = 30, col = "red3",linetype = "dashed", size = .5)+
  geom_hline(yintercept = 1, col = "red3",linetype = "dashed", size = .5)+
  annotate("text",x=45,data_plt[30,2]-.06,label=string)+
  annotate("rect",xmin=31,xmax=59,ymin=.91,ymax=.945,alpha=.3,fill="blue",color="black")+
  annotate("segment", x=30,xend=45,y=data_plt[30,2],yend=data_plt[30,2]-.043,color="red")+
  scale_x_discrete(name="Principal Component",limits=seq(0,85,by=5), breaks=seq(0,80,by=5))+
  scale_y_continuous(name ="Cumulative variance explained",limits=c(.37,1),breaks=seq(0,1,by=.1))
g1
ggsave("gg_pca_var_exp.png", plot=g1)
#theme(axis.text.x = element_text(size=14),
#     axis.text.y = element_text(size=14))
cor(train$critical_temp,train_pca$PC1)

#method = c("pearson", "kendall", "spearman") 

#________Plot_3_________mean critical temp
c=rbind(t(rep(0,86)),rep(0,86))
mean_crit = as.data.frame(c)
names(mean_crit)=names(uniq[,-c(88,87)])
uniq_test=uniq[,-c(88)]
size=dim(uniq_test)
for (ii in 1:(size[2]-1)) {
  nonZero=uniq_test[which(uniq_test[,ii] > 0),]
  meancrit=mean(nonZero$critical_temp)
  mean_crit[1,ii]=meancrit
  mean_crit[2,ii]=var(nonZero$critical_temp)
}
library(data.table)
mean_crit=rbind(mean_crit,colSums(uniq_test[,-87] != 0))
name=names(mean_crit)
meanTrans=as.data.frame(transpose(mean_crit))
meanTrans=cbind(meanTrans,name)
names(meanTrans)=c("crit","var","count","element")
meanTran=meanTrans[order(-meanTrans$crit),]
d=na.omit(meanTran)
d=cbind(d,1:dim(d)[1])
names(d)[5]="id"
library(ggrepel)
rownames(d) = 1:77 
#apply(uniq_test[,-87],MARGIN=2,function(z) length(z[z!=0]))
#equivalent to colSums(uniq_test[,-87] != 0)

g2=ggplot(d,aes(x=id,y=crit,label=element))+
  ggtitle("Elements vs. Average Critical Temperature")+
  scale_x_continuous(name="Element (labeled next to points)",limits=c(-.5,80),breaks=c(0,77),labels=c())+
  scale_y_continuous(name ="Critical Temperature (K)",limits=c(0,81),breaks=seq(5,80,by=5))+
  geom_point(aes(size=count,col=crit))+
  geom_label_repel(aes(label = element),
                   box.padding   = 0.25, 
                   point.padding = 0.05,
                   size = 3,
                   arrow = arrow(length = unit(0.06,"inches"), 
                                 type = "closed", ends = "first"),
                   force=9, segment.alpha=.7,
                   segment.color = 'grey15')+
  labs(size="Count element \n appearances",col="Temp.",
       caption="The critical Temperature is an average of the critical 
       temperature of all superconductor that element appears in.")+
  scale_colour_gradient(low = "black", high = "red",
                        name = waiver(),na.value="grey50",
                        limits=c(0,80), labels=seq(0,80,by=20),
                        breaks=seq(0,80,by=20))+
  #theme(legend.position = "bottom")+
  scale_size_continuous(breaks=seq(50,12000, by = 1500),
                        labels=seq(50,12000, by = 1500))
g2
ggsave("gg_ave_crit_count_elements4.png", plot=g2)


#________Plot_4_________mean critical temp
g3=ggplot(d,aes(x=id,y=crit,label=element))+
  ggtitle("Elements vs. Average Critical Temperature")+
  scale_x_continuous(name="Element (labeled next to points)",limits=c(-.5,80),breaks=c(0,77),labels=c())+
  scale_y_continuous(name ="Critical Temperature (K)",limits=c(0,81),breaks=seq(5,80,by=5))+
  geom_point(aes(size=count,col=var))+
  geom_label_repel(aes(label = element),
                   box.padding   = 0.25, 
                   point.padding = 0.05,
                   size = 3,
                   arrow = arrow(length = unit(0.06,"inches"), 
                                 type = "closed", ends = "first"),
                   force=9, segment.alpha=.7,
                   segment.color = 'grey15')+
  labs(size="Count element \n appearances",col="Variance",
       caption="The critical Temperature is an average of the critical 
       temperature of all superconductor that element appears in. The variance scale is the variance of the critical
       temeprature for the supercondudcotrs that contain that element")+
  scale_colour_gradient(low = "black", high = "yellow",
                        name = waiver(),na.value="grey50",
                        limits=c(0,2000), labels=seq(0,1800,by=200),
                        breaks=seq(0,1800,by=200))+
  #theme(legend.position = "bottom")+
  scale_size_continuous(breaks=seq(50,12000, by = 1500),
                        labels=seq(50,12000, by = 1500))

g3
ggsave("gg_ave_crit_var_elements3.png", plot=g3)


#_____________Plot_5____________hist of factor crit temp
#df_cut$critical_temp
g4=ggplot(df_cut, aes(x = critical_temp))+
  geom_bar(col="black", fill = "darkgreen", alpha = 0.4)+
  ggtitle("Distribution of Factorized Critical Temperature of Superconductors")+
  ylab("Count")
g4
ggsave("gg_fact_crit_temp_hist.png", plot=g4)

## ______ Plot 6 _______
g5=ggplot(as.data.frame(silstore),aes(x=1:10,y=silstore))+
  ggtitle("Clusters vs. Silhouette Score")+
  scale_x_continuous(name="Number of Clusters",limits=c(2,10),breaks=2:10,labels=2:10)+
  scale_y_continuous(name ="Silhouette Score",limits=c(.3,.55),breaks=seq(.3,.55,by=.05))+
  geom_point(size=3, col="darkblue")+
  geom_line(size=.5, col="darkred")
g5
ggsave("gg_clustersil.png", plot=g5)

