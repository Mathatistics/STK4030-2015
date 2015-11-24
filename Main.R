## ----setup, echo = FALSE-------------------------------------------------
library(knitr)
opts_knit$set(root.dir = '..')
opts_chunk$set(comment = NA)

## ----echo = FALSE, message = FALSE, error = TRUE, warning=FALSE----------
## Clear Everything
rm(list = ls())

## Load Sources and Packages
source('functions.R')
req.pkgs <- c('plyr', 'reshape2', 'ggplot2', 'glmnet', 'pls', 
              'parallel', 'MASS', 'pracma', 'data.table', 'klaR', 
              'splines', 'knitr', 'grid', 'gridExtra', 'base')
load.pkgs(req.pkgs)

## load and attach dataset
load('Data/wine.rdata')
load('Data/fossil.rdata')
load('Data/faces.rdata')
if (file.exists(file.path(getwd(), 'Exports', 'pcr.Rdata', fsep = '/'))) 
  load(file = 'Exports/pcr.Rdata')
if (file.exists(file.path(getwd(), 'Exports', 'pls.Rdata', fsep = '/')))
  load(file = 'Exports/pls.Rdata')
if (file.exists(file.path(getwd(), 'Exports', 'rda.Rdata', fsep = '/')))
  load(file = 'Exports/rda.Rdata')

## ----Q1, echo = FALSE, child = 'Question1.Rnw'---------------------------

## ----declare, echo = FALSE-----------------------------------------------
## Scaling the data
y <- as.matrix(wine[, "quality"])
colnames(y) <- 'quality'
X <- as.matrix(wine[, 1:11], ncol = 11)
colnames(X) <- names(wine)[1:11]
X <- scale(X) ## Scaling the covariates
test <- wine$test ## Declare Test or Train observations
## Make Model Equation Part ------------
get.eq.part.1 <- paste(paste('\\beta', 1:3, sep = '_'), 
                       paste('\\texttt{', names(wine)[1:3], '}', sep = ''), 
                       collapse = '+')
get.eq.part.2 <- paste(paste('\\beta', 4:6, sep = '_'), 
                       paste('\\texttt{', names(wine)[4:6], '}', sep = ''), 
                       collapse = '+')
get.eq.part.3 <- paste(paste('\\beta', 7:9, sep = '_'), 
                       paste('\\texttt{', names(wine)[7:9], '}', sep = ''), 
                       collapse = '+')
get.eq.part.4 <- paste(paste('\\beta', 9:11, sep = '_'), 
                       paste('\\texttt{', names(wine)[9:11], '}', sep = ''), 
                       collapse = '+')
## -----------

## ----wine.lm, echo=TRUE--------------------------------------------------
lm.wine.formula <- makeFormula(X, y)
lm.wine <- lm(lm.wine.formula, data = wine[!test, ])
summary(lm.wine)

## ----wine.test.lm.pred, echo = FALSE-------------------------------------
### Predict test sample
pred.wine.test <- predict(lm.wine, newdata = wine[test, ])
### Expected test error
wine.lm.test.err <- pracma::rmserr(pred.wine.test, y[test, ])
wine.lm.train.err <- pracma::rmserr(y[!test,], lm.wine$fitted.values)
### Covariate with strongest association
which.var.max.coef.p <- names(coef(lm.wine)[-1][which.max(coef(lm.wine)[-1])])
which.var.max.coef.n <- names(coef(lm.wine)[-1][which.min(coef(lm.wine)[-1])])

## ----wine.ridge, echo = TRUE, tidy=FALSE---------------------------------
rdg.wine <- cv.glmnet(X[!test, ], y[!test, ], alpha = 0, nfolds = 10)
rdg.wine.test.pred <- predict.cv.glmnet(object = rdg.wine, newx = X[test, ], 
                                    s = 'lambda.min')
rdg.wine.train.pred <- predict.cv.glmnet(object = rdg.wine, newx = X[!test, ], 
                                    s = 'lambda.min')

## ----wine.ridge.err, echo = FALSE----------------------------------------
wine.rdg.test.err <- rmserr(rdg.wine.test.pred, y[test, ])
wine.rdg.train.err <- rmserr(rdg.wine.train.pred, y[!test, ])

## ----wine.ridge.coef, echo = TRUE----------------------------------------
coef(rdg.wine)

## ----rdg.mse.lmc.plot, echo=FALSE----------------------------------------
### Preperation for plot
rdg.cv.wine.df <- data.frame(
  log.lmda = log10(rdg.wine$lambda),
  MSE = rdg.wine$cvm,
  upper = rdg.wine$cvup,
  lower = rdg.wine$cvlo
)
rdg.cv.range.df <- data.frame(
  lo = log10(rdg.wine$lambda.min),
  hi = log10(rdg.wine$lambda.1se)
)

### The plot log10(lambda) vs MSE
ridge.mse.lambda.plot <- ggplot(rdg.cv.wine.df, aes(log.lmda, MSE)) + 
  geom_point(color = 'red') + 
  geom_errorbar(aes(ymax = upper, ymin = lower)) + 
  theme_bw() + 
  labs(x = 'Log10(lambda)', y = 'Mean-squared Error') +
  geom_vline(data = rdg.cv.range.df, aes(xintercept = c(lo, hi)),
             color = 'blue',
             linetype = 2, size = 1)

## ----rdg.mse.lmc.plot.print, echo=FALSE, fig.width='0.8\textwidth', fig.height=4, fig.cap='Cross-validated mean squared error against the logarithm with base 10 of the tuning parameter $\\lambda$ values for Ridge Regression Model', fig.pos='H'----
print(ridge.mse.lambda.plot)

## ----wine.lso, echo = TRUE, tidy=FALSE-----------------------------------
lso.wine <- cv.glmnet(X[!test, ], y[!test, ], alpha = 1, nfolds = 10)
lso.wine.test.pred <- predict.cv.glmnet(object = lso.wine, newx = X[test, ], 
                                    s = 'lambda.min')
lso.wine.train.pred <- predict.cv.glmnet(object = lso.wine, newx = X[!test, ], 
                                    s = 'lambda.min')

## ----wine.lasso.err, echo = FALSE----------------------------------------
wine.lso.test.err <- rmserr(lso.wine.test.pred, y[test, ])
wine.lso.train.err <- rmserr(lso.wine.train.pred, y[!test, ])

## ----wine.lso.coef, echo = TRUE------------------------------------------
coef(lso.wine)

## ----lso.mse.lmc.plot, echo=FALSE----------------------------------------
### Preperation for plot
lso.cv.wine.df <- data.frame(
  log.lmda = log10(lso.wine$lambda),
  MSE = lso.wine$cvm,
  upper = lso.wine$cvup,
  lower = lso.wine$cvlo
)
lso.cv.range.df <- data.frame(
  lo = log10(lso.wine$lambda.min),
  hi = log10(lso.wine$lambda.1se)
)

### The plot log10(lambda) vs MSE
lso.mse.lambda.plot <- ggplot(lso.cv.wine.df, aes(log.lmda, MSE)) + 
  geom_point(color = 'red') + 
  geom_errorbar(aes(ymax = upper, ymin = lower)) + 
  theme_bw() + 
  labs(x = 'Log10(lambda)', y = 'Mean-squared Error') +
  geom_vline(data = lso.cv.range.df, aes(xintercept = c(lo, hi)),
             color = 'blue',
             linetype = 2, size = 1)

## ----lso.mse.lmc.plot.print, echo=FALSE, fig.width='0.8\textwidth', fig.height=4, fig.cap='Cross-validated mean squared error against the logarithm with base 10 of the tuning parameter $\\lambda$ values for Lasso Regression Model', fig.pos='H'----
print(lso.mse.lambda.plot)

## ----model.compare, echo=FALSE-------------------------------------------
test.err.dt <- data.table(melt(
  list(linear = wine.lm.test.err, 
       ridge = wine.rdg.test.err, 
       lasso = wine.lso.test.err)
))
setnames(test.err.dt, names(test.err.dt), c('TestError', 'ErrorType', 'Model'))
setkeyv(test.err.dt, c('Model', 'ErrorType'))
train.err.dt <- data.table(melt(
  list(linear = wine.lm.train.err, 
       ridge = wine.rdg.train.err, 
       lasso = wine.lso.train.err)
))
setnames(train.err.dt, names(train.err.dt), c('TrainError', 'ErrorType', 'Model'))
setkeyv(train.err.dt, c('Model', 'ErrorType'))

err.dt <- melt(test.err.dt[train.err.dt], 2:3, 
               variable.name = 'Which.Error', 
               value.name = 'Error')

mdl.comp <- ggplot(dplyr::filter(melt(list(linear = wine.lm.test.err, 
                                      ridge = wine.rdg.test.err, 
                                      lasso = wine.lso.test.err)), 
                                 L2 != 'nmse'), 
              aes(L2, value)) + 
  geom_line(aes(group = L1, color = L1)) + 
  geom_point(shape = 21, fill = 'gray') + 
  theme_bw() + 
  theme(legend.position = c(0.6, 0.15), 
        legend.title = element_blank(),
        legend.direction = 'horizontal',
        axis.title = element_blank(),
        legend.background = element_blank()) +
  geom_rug(side = 'l', aes(color = L1)) +
    annotate(geom = 'rect', xmin = 2.9, xmax = 3.1, 
             ymin = 0.53, ymax = 0.62, 
             alpha = 0, color = 'black')
msedf <- test.err.dt[ErrorType == 'mse']
mdl.comp.mse <- ggplot(msedf, aes(Model, TestError, color = Model)) + 
    geom_violin(size = 5) + theme_bw() + 
    theme(legend.position = 'none',
          axis.title = element_blank(),
          axis.text.y = element_blank(),
          axis.ticks.y = element_blank()) + 
    scale_y_continuous(limits = c(0.55, 0.59)) +
    labs(y = 'Mean Squared Error (MSE)') + 
    geom_rug(sides = 'l') +
    geom_text(aes(label = round(TestError, 3)), 
              vjust = -1, color = 'black', size = 4)

## ----model.compare.print, echo = FALSE, fig.width='\\textwidth', fig.height=2.7, fig.cap='Comparison of Linear, Ridge and Lasso model with respect to various error of test prediction with MSE shown in right side', fig.pos='H'----
gridExtra::grid.arrange(mdl.comp, mdl.comp.mse, ncol = 2, widths = c(7/10, 3/10))


## ----Q2, echo = FALSE, child = 'Question2.Rnw'---------------------------

## ----preparation, echo = FALSE-------------------------------------------
fossil <- data.table(fossil)
n.knots <- 40
n.order <- 4
## On generating 40 knots, the distance between them equals to 0.7613 ≈ 0.761
equal.knots <- seq(fossil[, min(age)], 
                   fossil[, max(age)], 
                   length.out = 42)[2:41]
## Quantile spaced knots is generated by bs function on specifying df

## ----spline.fit, echo = FALSE--------------------------------------------
bs.eql <- bs(x = fossil[, age],  df = n.knots + n.order + 1, 
             knots = equal.knots,  degree = n.order - 1, intercept = TRUE)
bs.qtl <- bs(x = fossil[, age], df = n.knots + n.order, 
             degree = n.order - 1, intercept = TRUE)

## Fitting Model
eql.knot.model <- lm(strontium.ratio ~ bs.eql, data = fossil)
qtl.knot.model <- lm(strontium.ratio ~ bs.qtl, data = fossil)

pred.dt <- melt(
  data.table(Age = fossil[, age],
             Strontium.Ratio = fossil[, strontium.ratio],
             Equal.Knots.Prediction = eql.knot.model$fitted.values,
             Quantile.Knots.Prediction = qtl.knot.model$fitted.values),
  id.vars = 1:2)

predplot <- ggplot(pred.dt, aes(Age, Strontium.Ratio)) + 
  geom_point() + 
  geom_line(aes(color = variable, y = value)) + 
  theme_bw() + 
  theme(legend.title = element_blank(), 
        legend.position = 'top')

## ----basisPlot, echo = TRUE, fig.cap='B-splines for 1000 \\texttt{age} samples sequenced between its range. These B-splines have 40 internal knots located and have boundry at the range of \\texttt{age}', fig.subcap=c("Knots equidistanced with 0.761 distance apart", "Knots located at the quantiles of distribution of \\texttt{age}"), fig.width='\\textwidth', fig.show='hold', fig.height=1.7, fig.pos='H'----
plot(bs.eql); plot(bs.qtl)

## ----pred.spline.plot, echo = FALSE, fig.cap='Prediction from the B-spline function with equally spaced knots and knots located at quantiles of the distribution of predictor variable age', fig.height=3.5, fig.pos='H'----
print(predplot)

## ----splineFit, echo=FALSE, fig.pos='H', fig.cap='Spline Smooth curve fitted to the fossil data with the smoothing parameter obtained from leave-one-out cross-validation and Generalized cross-validation method', fig.height=3.5----
spline.cv <- smooth.spline(x = fossil[, age],
                           y = fossil[, strontium.ratio],
                           cv = TRUE,
                           nknots = 40,
                           all.knots = FALSE)
spline.gcv <- smooth.spline(x = fossil[, age],
                           y = fossil[, strontium.ratio],
                           cv = FALSE,
                           nknots = 40,
                           all.knots = FALSE)
spline.fit <- data.table(Age = fossil[, age],
                         Strontium.Ratio = fossil[, strontium.ratio],
                         spline.fit.age = spline.cv$x,
                         Strontium.Ratio.CV = spline.cv$y,
                         Strontium.Ratio.GCV = spline.gcv$y)
newx <- 113.5
newYpred <- data.frame(x = newx, 
                       y = c(predict(spline.cv, x = newx)$y, 
                             predict(spline.gcv, x = newx)$y))

                            
splineplot <- ggplot(melt(spline.fit, 1:3), aes(Age, Strontium.Ratio)) +
  geom_point() + 
  geom_line(aes(x = spline.fit.age, 
                y = value, color = variable)) +
  theme_bw() + theme(legend.title = element_blank(), legend.position = 'top') +
  geom_hline(yintercept = newYpred$y, col = 'gray', linetype = 2, size = 0.25) +
  geom_vline(xintercept = newYpred$x, col = 'gray', linetype = 2, size = 0.25) +
  annotate(geom = 'text', newYpred$x, newYpred$y, 
           label = round(newYpred$y, 4), 
           vjust = -3, hjust = -0.1, size = 4, color = 'blue', bg = 'white') +
  geom_point(data = newYpred, aes(x = x, y = y), 
             fill = 'red', size = 2.5, shape = 24)
print(splineplot)


## ----Q3, echo = FALSE, child = 'Question3.Rnw'---------------------------

## ----attr.table, echo=FALSE----------------------------------------------
## Using DataTable
faces.dt <- data.table(faces)
## Create Sex Logical Variable -------------------------------------------------
Male <- rep(c(TRUE, FALSE), each = 100)
shoulder <- as.logical(shoulder)
attr.dt <- data.table(
  Gender = factor(Male, 
                  levels = c('TRUE', 'FALSE'), 
                  labels = c('Male', 'Female')),
  Shoulder = factor(shoulder, 
                  levels = c('TRUE', 'FALSE'), 
                  labels = c('With Shoulder', 
                             'Without Shoulder'))
  )
## Factor Combining to create new one
invisible(attr.dt[, Gender.Shoulder := dae::fac.combine(list(
  attr.dt[, Gender], attr.dt[, Shoulder]), combine.levels = T)
  ])

## ----avg.face.img, echo=FALSE, fig.height=4.3, fig.width=8.2, fig.cap='Average (Mean) of portraits for Male and Female', fig.pos='H'----
## Plot average faces
avgFaces <- faces.dt[, list(AvgMale = rowMeans(.SD[, Male, with = F]),
                         AvgFemale = rowMeans(.SD[, !Male, with = F]))]
getFaced(as.matrix(avgFaces), n.faces = 1:2, facet.ncol = 2)

## ----face.pca, echo=TRUE-------------------------------------------------
pc.a <- prcomp(t(faces))

## ----face.pca.output, echo = FALSE---------------------------------------
pc.score <- data.table(pc.a$x)
pc.rotate <- data.table(pc.a$rotation)

## ----pca.eigenfaces, echo=FALSE, fig.height=2.8, fig.cap='Eigen faces obtained from first three principal component for all 200 faces including both male and female', fig.pos = 'H'----
getFaced(as.matrix(pc.rotate), n.faces = 1:3, facet.ncol = 3)

## ----Scoreplot, echo=FALSE, fig.cap='Principal components plots (Score plots) obtained from pca analysis colored according to Gender and Shoulder being present or not', fig.pos='H', fig.height=6.5----
listGrid <- expand.grid(list(c(1,2), c(1,3)), c('Gender', 'Shoulder'))
score.plt <- mlply(listGrid, function(Var1, Var2){
  Var1 <- unlist(Var1); Var2 <- as.character(Var2)
  getScored(pc.score, ncomp = 1:3, 
            which = Var1, 
            attr.df = attr.dt, 
            col.var = Var2)
})
getGrided(score.plt)

## ----create.response, echo=FALSE-----------------------------------------
## Creating Gender Response -------
Gender <- ifelse(Male, -1, 1)
Shoulder <- ifelse(shoulder, -1, 1)
Gender.Shoulder <- attr.dt[, as.numeric(Gender.Shoulder)]

## ----faces.model.fitting, echo = FALSE-----------------------------------
if (!('pc.r' %in% ls()) | !('pls.r' %in% ls()))
  pls.options(parallel = makeCluster(6, type = "PSOCK"))
if (!('pc.r' %in% ls())) {
  pc.r <- pcr(Gender ~ t(faces), validation = 'LOO')
  save(pc.r, file = 'Exports/pcr.Rdata')
}
if (!('pls.r') %in% ls()) {
  pls.r <- plsr(Gender ~ t(faces), validation = 'LOO')
  save(pls.r, file = 'Exports/pls.Rdata')
}
if (!('pc.r' %in% ls()) | !('pls.r' %in% ls()))
  stopCluster(pls.options()$parallel)

## ----faces.msep.pcr, echo = FALSE----------------------------------------
pcr.msep <- adply(MSEP(pc.r)$val, 3)[-1, ]
pcr.msep[, 1] <- as.numeric(pcr.msep[, 1]) - 1

pcr.min.comp <- which.min(pcr.msep$adjCV)

## ----faces.msep.pcr.plot, echo = FALSE, fig.height=3.5, fig.cap='Number of principal component against Root Mean Square Error of Prediction plot for Principal Component Regression. The dashed line shows the number of component needed for minimum error', fig.pos='H'----
plotRMSEP(MSEP(pc.r))

## ----pcr.conf.plot, echo=FALSE, fig.cap='Confusion Plot for PCR classification', fig.height=2, fig.pos = 'H'----
getClassified(pc.r$fitted.values[, , pcr.min.comp], Gender)$conf.plot

## ----faces.msep.pls, echo = FALSE----------------------------------------
pls.msep <- adply(MSEP(pls.r)$val, 3)[-1, ]
pls.msep[, 1] <- as.numeric(pls.msep[, 1]) - 1

pls.min.comp <- which.min(pls.msep$adjCV)

## ----faces.msep.pls.plot, echo = FALSE, fig.height=3.5, fig.cap='Number of principal component against Root Mean Square Error of Prediction plot for Partial Least Squre regression. The dashed line shows the number of component needed for minimum error', fig.pos='H'----
plotRMSEP(MSEP(pls.r))

## ----pls.conf.plot, echo=FALSE, fig.cap='Confusion Plot for PLS classification', fig.height=2, fig.pos = 'H'----
getClassified(pls.r$fitted.values[, , pls.min.comp], Gender)$conf.plot

## ----qdaSetup, echo = FALSE----------------------------------------------
## Quadratic Decision Analysis
pc.model.mat <- data.frame(pc.score, attr.dt)
qda.md.name <- c('Gender', 'Shoulder', 'Gender.Shoulder')
qda.fit <- llply(qda.md.name, function(x){
  qda(resp ~ ., data = data.frame(resp = eval(parse(text = x)), 
                                  pc.score[, 1:5, with = F]))
})
names(qda.fit) <- qda.md.name

## ----qdaModel, echo = FALSE----------------------------------------------
qda.score.plot <- mlply(listGrid, function(Var1, Var2) {
    Var1 <- unlist(Var1); Var2 <- as.character(Var2)
    if (Var2 == 'Gender')
      qdb <- getDB(pc.score, grid.size = 25, 
                   da.fit = qda.fit$Gender, n.comp = Var1)
    if (Var2 == 'Shoulder')
      qdb <- getDB(pc.score, grid.size = 25, 
                   da.fit = qda.fit$Shoulder, n.comp = Var1)
    plt <- getScored(pc.score, ncomp = 1:3, 
                     which = Var1, 
                     attr.df = attr.dt, 
                     col.var = Var2) 
    plt <- plt + geom_contour(data = qdb, 
                              aes_string(paste('PC', Var1, sep = ''), z = 'z'), 
                              bins = 1)
    return(plt)
  })
cf.df.qda.gender <- table(predict(qda.fit$Gender)$class, Gender)
msc.rate.qda <- 1 - sum(diag(cf.df.qda.gender))/sum(cf.df.qda.gender)

## ----qdaPlotGenderPrint, echo = FALSE, fig.cap='Classification of Gender from the \\texttt{faces} data. The blue line is the decision boundry obtained from QDA.', fig.pos = 'H', fig.height=4----
qda.score.plot$ncol <- 2
do.call(grid_arrange_shared_legend, qda.score.plot[c(1:2, 5)])

## ----qdaPlotShoulderPrint, echo = FALSE, fig.cap='Classification of a person with shoulder and without shoulder from the \\texttt{faces} data. The blue line is the decision boundry obtained from QDA.', fig.pos = 'H', fig.height=4----
do.call(grid_arrange_shared_legend, qda.score.plot[3:5])

## ----modifiedQDA, echo = FALSE-------------------------------------------
## Question 3(g) ------------------------
qda.gs.plot <- llply(list(c(1,2), c(1,3)), function(x){
  GS.db <- getDB(pc.score, grid.size = 25, da.fit = qda.fit$Gender.Shoulder, 
                 n.comp = x)
  plt <- getScored(scores(pc.a), ncomp = 1:3, 
                   which = x, 
                   attr.df = attr.dt, 
                   col.var = 'Gender.Shoulder')
  plt <- plt + geom_contour(data = GS.db, aes_string(names(GS.db)[1:2], z = 'z'),
                            lineend = 'round', linejoin = 'round', linetype = 1,
                            bins = 3)
  return(plt)
})
qda.gs.plot$ncol <- 2

## Classifications
qda.gs.hat <- predict(qda.fit$Gender.Shoulder)$class
qda.gs.hat <- factor(qda.gs.hat, levels = 1:4, 
                     labels = levels(attr.dt$Gender.Shoulder))
cf.gs.tbl <- table(attr.dt$Gender.Shoulder, qda.gs.hat)
error.rate <- 1 - sum(diag(cf.gs.tbl)) / sum(cf.gs.tbl)

attr.plus <- data.frame(attr.dt, Gender.Shoulder.Fitted = qda.gs.hat)

## Merging within Gender
qda.gender.fit2 <- factor(ifelse(grepl('Male', qda.gs.hat), 'Male', 'Female'), 
                          levels = c('Male', 'Female'))
cf.qda.gender.tbl <- table(qda.gender.fit2, attr.dt$Gender)
msc.rate.qda2 <- 1 - sum(diag(cf.qda.gender.tbl)) / sum(cf.qda.gender.tbl)

## ----modifiedQDAplt, echo = FALSE, fig.height=4, fig.cap='Decision Boundry for QDA on merged factor of gender and presence or absence of shoulder', fig.pos='H'----
do.call(grid_arrange_shared_legend, qda.gs.plot)

## ----modifiedQDA2, echo = FALSE, results='hide'--------------------------
qda.gs.plot2 <- llply(list(c(1,2), c(1,3)), function(x){
    plt <- getScored(scores(pc.a), ncomp = 1:3, 
                     which = x, 
                     attr.df = data.table(Gender = qda.gender.fit2), 
                     col.var = 'Gender')
    return(plt)
})
qda.gs.plot2$ncol <- 2

## ----modifiedQDAplt2, echo = FALSE, fig.height=3.5, fig.cap='Decision Boundry for QDA on merged factor of persion having shoulder or not within gender variable as if it was classification for just gender.', fig.pos = 'H'----
do.call(grid.arrange, qda.gs.plot2)

## ----rda, echo = FALSE---------------------------------------------------
if (!('rda.fit' %in% ls())) {
  rda.fit <- rda(resp ~ ., 
                 data = data.frame(resp = Gender, 
                                   pc.score[, 1:50, with = F]),
                 regularization = c(lambda = 0, gamma = 'gamma'),
                 crossval = TRUE, 
                 fold = nrow(pc.score))
  save(rda.fit, file = 'Exports/rda.Rdata')
}

## ----rdaPlot, echo = FALSE-----------------------------------------------
rda.gs.plot <- llply(list(c(1,2), c(1,3)), function(x){
    GS.db <- getDB(pc.score, grid.size = 50, da.fit = rda.fit, 
                   n.comp = x)
    plt <- getScored(scores(pc.a), ncomp = 1:3, 
                     which = x, 
                     attr.df = attr.dt, 
                     col.var = 'Gender')
    plt <- plt + geom_contour(data = GS.db, aes_string(names(GS.db)[1:2], z = 'z'),
                              lineend = 'round', linejoin = 'round', linetype = 1,
                              bins = 1)
    return(plt)
})
rda.gs.plot$ncol <- 2

## ----rdaPlotPrint, echo=FALSE, fig.cap='Classification of gender with decision boundry from Regularized Discreminant Analysis setup for scores of PC1 plotted against PC2 and PC3', fig.height=3.5, fig.pos='H'----
do.call(grid_arrange_shared_legend, rda.gs.plot)


## ----rcodes, eval = FALSE, ref.label=knitr::all_labels()[-3], results='markup', tidy=FALSE----
## NA

