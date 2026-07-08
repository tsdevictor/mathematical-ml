library(dplyr)
library(ggplot2)

n = 1000
X = matrix(0, nrow=n, ncol=2)
for (i in 1:n) {
  X[i,1] = runif(1, -4, 4)
  X[i,2] = runif(1, -4, 4)
}

y = 1*(X[,2] +2 > exp(.1*X[,1]**2 + sin(X[,2])))
df = data.frame(x1=X[,1], x2=X[,2],  y=y)
g = ggplot(data=df) + geom_point(aes(x=x1, y=x2, color=factor(y)))
print(g)

write.csv(df, "dataset.csv", row.names=FALSE)
