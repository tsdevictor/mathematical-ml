library(dplyr)
library(ggplot2)

separable = F
if (separable) {
  mu = 10
} else {
  mu = 1.5
}

n = 1000
X1 = matrix(rnorm(2*n), nrow=n, ncol=2)
X2 = matrix(mu + rnorm(2*n), nrow=n, ncol=2)
y = c(rep(-1, n), rep(1, n))
X = rbind(X1, X2)

ind = sample.int(2*n)
X = X[ind,]
y = y[ind]

df = data.frame(x1=X[,1], x2=X[,2],  y=y)
g = ggplot(data=df) + geom_point(aes(x=x1, y=x2, color=factor(y)))
print(g)

if (separable) {
  write.csv(df, "dataset_separable.csv", row.names=FALSE)
} else {
  write.csv(df, "dataset_non_separable.csv", row.names=FALSE)
}
