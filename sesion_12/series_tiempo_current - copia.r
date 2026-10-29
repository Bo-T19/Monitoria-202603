if (!require('forecast')) install.packages('forecast')
if (!require('tseries')) install.packages('tseries')
if (!require('fpp')) install.packages('fpp')

if (!require('tsoutliers')) install.packages('tsoutliers')


## Lectura del archivo  
## Revisar que el archivo se encuentre en documentos, 
## o la ruta en donde se encuentra direccionado R

getwd()
library(readxl)


#festivos<-read_xlsx("")
festivos<-read_xlsx("C:/Users/santi/OneDrive - Pontificia Universidad Javeriana/Maestría/Semestre 1/Métodos y Aplicaciones de Analítica/Archivos de trabajo/fiestaseasonal.xlsx"
)


producto1 <- ts(festivos$FIESTA1, start=c(2009, 1), end=c(2011, 12), frequency=12)
producto1.train <- window(producto1, start=c(2009, 1), end=c(2010, 12))
producto1.test <- window(producto1, start=c(2011, 1), end=c(2011, 12))
modelo1<-ets(producto1.train, model="ZZZ",opt.crit="mse",ic="aic")
summary(modelo1)
modelo2<-auto.arima(producto1.train)
summary(modelo2)
modelo.test<-ets(producto1.test,model=modelo1,use.initial.values = TRUE)
modeloar.test<-Arima(producto1.test, model=modelo2, use.initial.values=TRUE)
accuracy(modelo.test)
accuracy(modeloar.test)



# revision de la estructura y ver los primeros 3 registros del conjunto de datos
str(festivos)
head(festivos,3)


#--------AQUÍ COMIENZA EN SERIO----------#

#Construir para cada serie un objeto tipo serie de tiempo. 
producto1 <- ts(festivos$FIESTA1, start=c(2007, 1), end=c(2014, 2), frequency=12)
producto2 <- ts(festivos$FIESTA2, start=c(2007, 1), end=c(2014, 2), frequency=12)

#por supuesto puedo tenerlos todos a una
productos<-ts(festivos[1:4],start=c(2009, 1), end=c(2011, 2), frequency=12)


# realizar una muestra de entrenamiento (Enero 2007 a Diciembre 2012)
producto1.train <- window(producto1, start=c(2007, 1), end=c(2012, 12))
producto2.train <- window(producto2, start=c(2007, 1), end=c(2012, 12))

# realizar una muestra de validación (Enero 2013 a Febrero 2014)
producto1.test <- window(producto1, start=c(2013, 1), end=c(2014, 2))

plot(producto1.train)

#términos diferenciados
plot(diff(producto1.train, differences=1))
#el lag me permite hacerlo estacional
plot(diff(producto1.train, lag=12, differences=1))

Acf(producto1.train)
Pacf(producto1.train)

# eliminando el efecto estacional
Acf(diff(producto1.train, lag=12, differences=1))
Pacf(diff(producto1.train, lag=12, differences=1))

#creo desplazamientos
ar1<-lag(producto1.train,-1)
ar2<-lag(producto1.train,-2)
ar3<-lag(producto1.train,-3)
ar4<-lag(producto1.train,-4)
#los uno
base<-as.data.frame(cbind(producto1.train,ar1,ar2,ar3,ar4))
#dejo solo los datos completos
autocormatrix<-base[5:72,]
#obtengo la matriz de autocorrelaciones para k=5
cor(autocormatrix[,1:4])
#comparo con el gráfico
Acf(producto1.train)

#una forma de entender las autocorrelaciones parciales
parciales<-lm(producto1.train~ar1+ar2,data=autocormatrix)
coefficients(parciales)
parciales2<-lm(producto1.train~ar1+ar2+ar3,data=autocormatrix)
coefficients(parciales2)
parciales3<-lm(producto1.train~.,data=autocormatrix)
coefficients(parciales3)
summary(parciales3)


#modelo
modelo1<-ets(producto1.train, model="ZZZ",opt.crit="mse",ic="aic")
summary(modelo1)

#auto.arima busca el mejor arima. 
modelo2<-auto.arima(producto1.train)
summary(modelo2)

modelo3<-auto.arima(producto1.train,xreg=producto2.train)
summary(modelo3)

modelo4<-naive(producto1.train)
summary(modelo4)

adf.test(residuals(modelo1))
Box.test(residuals(modelo1), lag=20, type="Ljung-Box")

adf.test(residuals(modelo2))
Box.test(residuals(modelo2), lag=20, type="Ljung-Box")

plot(modelo1)
plot(forecast(modelo1,h=14,level=95))
plot(forecast(modelo2,h=14,level=95))

modelo.test<-ets(producto1.test,model=modelo1,use.initial.values = TRUE)
modeloar.test<-Arima(producto1.test, model=modelo2, use.initial.values=TRUE)

accuracy(modelo.test)
accuracy(modeloar.test)
modelo2$arma
?Arima
far2 <- function(x,h){forecast(Arima(producto1, order=c(0,0,0),seasonal=c(2,1,0)), h=h)}
errores<-tsCV(producto1,far2,h=12,initial=24)
#también se pueden usar ventanas móviles de tamaño k usando window=k
erroresd<-as.data.frame(errores)[25:86,]
erroresd2<-(erroresd)^2

RMSEs<-apply(erroresd2,2,function(y) sqrt(mean(y,na.rm=TRUE)))
RMSEs
mean(RMSEs,na.rm=TRUE)
RMSEs

far3 <- function(x,h){forecast(ets(producto1.train, model="MNM",opt.crit="mse",ic="aic"), h=h)}
errores2<-tsCV(producto1,far3,h=12,initial=24)
#también se pueden usar ventanas móviles de tamaño k usando window=k
erroresda<-as.data.frame(errores2)[25:86,]
erroresd2a<-(erroresda)^2

RMSEsa<-apply(erroresd2a,2,function(y) sqrt(mean(y,na.rm=TRUE)))
mean(RMSEsa,na.rm=TRUE)
RMSEsa



result<-tso(producto1.train)
#estos son los outliers
result$outliers
#Muestra los coeficientes, incluido los regresores de los outliers para quitarlos
result$fit$coef
?tso
result$fit$
result$fit$arma

result$fit$fitted

"A compact form of the specification, as a vector giving the number of AR, MA,
seasonal AR and seasonal MA coefficients, 
plus the period and the number of non-seasonal and seasonal differences"

