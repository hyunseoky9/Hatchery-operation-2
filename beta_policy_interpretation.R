# uses the spring transition data from 'results copy.ipynb'to conduct beta regression of production proportion against 
# different predictors.
# Run the cell generating simulation_spring_transitions_seed....csv from the jupyter notebook first using fpidx=3 ([813438,122,5])
#install.packages("betareg")
rm(lis=ls())
library(betareg)

filename ="G:/My Drive/research/nmsu/hatchery operation/codes/dynamic programming2/manuscript_results/simulation_spring_transitions_seed813438_paramset122_c5_Hatchery3.3.7.csv"
df = read.csv(filename)

# prepare response
#production has 0 or 1's and they need to be transofrmed (Smithson & Verkuilen, 2006)
idx1 = which(df$production == 1)
idx0 = which(df$production == 0)
df$production_tr = (df$production + 0.001)/1.002
(length(idx1) + length(idx0))/length(df$production)
prod = df$production_tr
# prepare predictors
logcatch_apr = log10(df$catch_total_apr.oct.nov+1)
logcatch_jul = log10(df$catch_total_jul.aug.sep+1)
logmincatch_apr = log10(df$min_catch_apr.oct.nov_reach+1)
logmincatch_jul = log10(df$min_catch_jul.aug.sep_reach+1)
q = df$q_kaf
gini = df$gini
cv = df$CV

df2 = data.frame(prod,logcatch_apr,logcatch_jul,logmincatch_apr,logmincatch_jul, logmincatch,q,cv)

df2[, c('logcatch_apr','logcatch_jul','logmincatch_apr','logmincatch_jul','logmincatch',"q","cv")] <-
  scale(df2[, c('logcatch_apr','logcatch_jul','logmincatch_apr','logmincatch_jul', 'logmincatch',"q","cv")])

cor(df2)
model0 <- betareg(
    prod ~ 
    logcatch_apr +
    logmincatch_apr +
    q+
    cv,
  data =df2,
  link = "logit"
)

model1 <- betareg(
  prod ~ 
    logcatch_apr +
    logmincatch_apr +
    q+
    cv| q,
  data =df2,
  link = "logit"
)

model2 <- betareg(
  prod ~ 
    logcatch_apr +
    logmincatch_apr +
    q+
    cv| logcatch_apr,
  data =df2,
  link = "logit"
)

model3 <- betareg(
  prod ~ 
    logcatch_apr +
    logmincatch_apr +
    q+
    cv| logmincatch_apr,
  data =df2,
  link = "logit"
)

model4 <- betareg(
  prod ~ 
    logcatch_apr +
    logmincatch_apr +
    q+
    cv| logcatch_apr + q,
  data =df2,
  link = "logit"
)

model5 <- betareg(
  prod ~ 
    logcatch_apr +
    logmincatch_apr +
    q+
    cv| logcatch_apr + logmincatch_apr,
  data =df2,
  link = "logit"
)

model6 <- betareg(
  prod ~ 
    logcatch_apr +
    logmincatch_apr +
    q+
    cv| logmincatch_apr + q,
  data =df2,
  link = "logit"
)

model7 <- betareg(
  prod ~ 
    logcatch_apr +
    logmincatch_apr +
    q+
    cv| logmincatch_apr + q + cv,
  data =df2,
  link = "logit"
)

model7 <- betareg(
  prod ~ 
    logcatch_apr +
    logmincatch_apr +
    q
    | logmincatch_apr+ q,
  data =df2,
  link = "logit"
)


AIC(model0,model1,model2,model3,model4,model5,model6,model7)
summary(model7)

# model 7 has least number of predictors with 2nd highest AIC with <2 difference from the best AIC



