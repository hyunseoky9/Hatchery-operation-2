# dirichlet regression of stocking action for RL's trained on 
# 3.3.7 (monitoring observation)
library(DirichletReg)
library(dplyr)
library(readr)
library(Formula)
library(ggtern)
library(ggrepel)
library(dplyr)
library(purrr)
# 1.  performs Dirichlet regression on simulated data using TD3 policy with c=4
# where response is the stocking distribution accross 3 reaches, and covariates are 
# engineered features of state variables in the hatchery environment
# 2. With the selected regression model, find 2 covariates that explains the policy the best
# using drop-one method and comparing the change in AIC. 
# 3. draw a ternary effect plot

# 1. Dirichlet regression and choosing best model
rm(list=ls())
params = c(813438,122,5)
filename = sprintf("G:/My Drive/research/nmsu/hatchery operation/codes/dynamic programming2/manuscript_results/simulation_fall_transitions_seed%d_paramset%d_c%d_Hatchery3.3.7.csv",params[1],params[2],params[3])
falldf_analysis <- read.csv(filename)
falldf_analysis$Y <- DR_data(falldf_analysis[,c('stock_a','stock_i','stock_s')])


# models
mod6.2 <- DirichReg(Y ~ log_catch_jul.aug.sep_a + log_catch_jul.aug.sep_i + log_catch_jul.aug.sep_s + lowcatch_jul.aug.sep_a+ lowcatch_jul.aug.sep_i+ lowcatch_jul.aug.sep_s+ 
                      gini+ relcatchshare_jul.aug.sep_a + relcatchshare_jul.aug.sep_i,
                    falldf_analysis,model='alternative')  

mod7.2 <- DirichReg(Y ~ log_catch_jul.aug.sep_a+ log_catch_jul.aug.sep_i+ log_catch_jul.aug.sep_s+ lowcatch_jul.aug.sep_a+ lowcatch_jul.aug.sep_i+ lowcatch_jul.aug.sep_s+ 
                      relcatchshare_jul.aug.sep_a + relcatchshare_jul.aug.sep_i,
                    falldf_analysis,model='alternative')  

mod8.2 <- DirichReg(Y ~ log_catch_jul.aug.sep_a + log_catch_jul.aug.sep_i+ log_catch_jul.aug.sep_s+ nocatch_jul.aug.sep_a+ nocatch_jul.aug.sep_i+nocatch_jul.aug.sep_s+
                      lowcatch_jul.aug.sep_a+ lowcatch_jul.aug.sep_i+ lowcatch_jul.aug.sep_s+
                      relcatchshare_jul.aug.sep_a + relcatchshare_jul.aug.sep_i,
                    falldf_analysis,model='alternative')  

mod9.2 <- DirichReg(Y ~ log_catch_jul.aug.sep_a+ log_catch_jul.aug.sep_i+ log_catch_jul.aug.sep_s+
                      relcatchshare_jul.aug.sep_a + relcatchshare_jul.aug.sep_i,
                    falldf_analysis,model='alternative')  

mod10.2 <- DirichReg(Y ~ log_catch_jul.aug.sep_a+ log_catch_jul.aug.sep_i+ log_catch_jul.aug.sep_s+
                       nocatch_jul.aug.sep_a+ nocatch_jul.aug.sep_i+ nocatch_jul.aug.sep_s,
                     falldf_analysis,model='alternative')  

mod11.2 <- DirichReg(Y ~ log_catch_jul.aug.sep_a+ log_catch_jul.aug.sep_i+ log_catch_jul.aug.sep_s,
                     falldf_analysis,model='alternative')  

# standardized version of mod9.2
falldf_analysis_std <- falldf_analysis %>%
  mutate(
    log_catch_jul.aug.sep_a_std = scale(log_catch_jul.aug.sep_a),
    log_catch_jul.aug.sep_i_std = scale(log_catch_jul.aug.sep_i),
    log_catch_jul.aug.sep_s_std = scale(log_catch_jul.aug.sep_s),
    relcatchshare_jul.aug.sep_a_std    = scale(relcatchshare_jul.aug.sep_a),
    relcatchshare_jul.aug.sep_i_std    = scale(relcatchshare_jul.aug.sep_i)
  )

mod9.2_standardized <- DirichReg(
  Y ~ log_catch_jul.aug.sep_a_std + log_catch_jul.aug.sep_i_std + log_catch_jul.aug.sep_s_std +
    relcatchshare_jul.aug.sep_a_std + relcatchshare_jul.aug.sep_i_std,
  data = falldf_analysis_std,
  model = "alternative"
)

AIC(mod8.2)
AIC(mod9.2)
AIC(mod10.2)
AIC(mod11.2)
summary(mod11.2)
summary(mod9.2_standardized)
pred_mod9.2 = predict(mod9.2)


# plot obs vs pred
# congruence analysis (in SI2)
obs = falldf_analysis[,c('stock_a','stock_i','stock_s')]
plottingmod = mod11.2
plottingmod_pred = predict(plottingmod)
congruence = pmin(plottingmod_pred,as.matrix(obs))
mean(apply(congruence,1,sum)) # congruence

if(1==1)
{par(mfrow=c(1,3))
  plot(plottingmod_pred[,1],obs$stock_a,main=sprintf('Angostura corr: %.2f',cor(plottingmod_pred[,1],obs$stock_a)),xlim=c(0,1),ylim=c(0,1))
  abline(a=0, b=1, col="red", lwd=2)
  plot(plottingmod_pred[,2],obs$stock_i,main=sprintf('Isleta corr: %.2f',cor(plottingmod_pred[,2],obs$stock_i)),xlim=c(0,1),ylim=c(0,1))
  abline(a=0, b=1, col="red", lwd=2)
  plot(plottingmod_pred[,3],obs$stock_s,main=sprintf('San Acacia corr: %.2f',cor(plottingmod_pred[,3],obs$stock_s)),xlim=c(0,1),ylim=c(0,1))
  abline(a=0, b=1, col="red", lwd=2)
}

# MODEL OF CHOICE : 9.2
mod = mod9.2
pred = pred_mod9.2

# 2. Drop-one analysis to find 2 covariates that explain the variation in response the most.
if(1==0)
{
  base_aic = AIC(mod)
  vars <- attr(terms(mod), "term.labels")
  deltaAIC <- sapply(vars, function(v) {
    m2 <- update(mod, as.Formula(sprintf("Y ~ . - %s | .", v)))
    AIC(m2) - base_aic
  })
  sort(deltaAIC)
  
  # log_popsize_i and log_popsize_s are the two continuous variables that changes AIC the most.
  # * Look for plotting of the stocking distribution with these two variables. 
  
  # 2.2 Drop-one analysis by looking at MSE for each reach.
  # angostura
  vars <- attr(terms(mod), "term.labels")
  base_prop_a = falldf_analysis$stock_a
  base_prop_i = falldf_analysis$stock_i
  base_prop_s = falldf_analysis$stock_s
  base_MSE_a = sum((pred[,1] - falldf_analysis$stock_a)^2)
  base_MSE_i = sum((pred[,2] - falldf_analysis$stock_i)^2)
  base_MSE_s = sum((pred[,3] - falldf_analysis$stock_s)^2)
  deltaMSE_a <- sapply(vars, function(v) {
    m2 <- update(mod, as.Formula(sprintf("Y ~ . - %s | .", v)))
    pred2 <- predict(m2)
    m2MSE <- sum((pred2[,1] - base_prop_a)^2)
    m2MSE - base_MSE_a
  })
  sort(deltaMSE_a)
  deltaMSE_i <- sapply(vars, function(v) {
    m2 <- update(mod, as.Formula(sprintf("Y ~ . - %s | .", v)))
    pred2 <- predict(m2)
    m2MSE <- sum((pred2[,2] - base_prop_i)^2)
    m2MSE - base_MSE_i
  })
  sort(deltaMSE_i)
  deltaMSE_s <- sapply(vars, function(v) {
    m2 <- update(mod, as.Formula(sprintf("Y ~ . - %s | .", v)))
    pred2 <- predict(m2)
    m2MSE <- sum((pred2[,3] - base_prop_s)^2)
    m2MSE - base_MSE_s
  })
  sort(deltaMSE_s)
}

# 3. Ternary effect plot


# ---- helper: find a representative
# typical values (median for numeric, majority for binary 0/1) ----
{
  typical_row <- function(df, exclude = c("stock_a","stock_i","stock_s")){
    X <- df[, setdiff(names(df), exclude), drop = FALSE]
    out <- lapply(X, function(x){
      if(is.numeric(x)){
        ux <- sort(unique(x))
        if(length(ux[!is.na(ux)]) == 2 && all(ux %in% c(0,1))) {
          as.numeric(names(sort(table(x), decreasing=TRUE)[1]))  # mode 0/1
        } else median(x, na.rm=TRUE)
      } else {
        names(sort(table(x), decreasing=TRUE))[1]
      }
    })
    as.data.frame(out, stringsAsFactors = FALSE)
  }
  
  typical_row(falldf_analysis)
  
  # pick top-K predictors by ΔAIC (mean side) 
  top_k_predictors <- function(fit, data, K = 8){
    mean_terms <- attr(terms(formula(fit), rhs = 1), "term.labels")
    base_AIC <- AIC(fit)
    dAIC <- sapply(mean_terms, function(v){
      m2 <- suppressWarnings(update(fit, as.Formula(sprintf("Y ~ . - %s | .", v))))
      AIC(m2) - base_AIC
    })
    tibble(term = names(dAIC), dAIC = as.numeric(dAIC)) %>%
      arrange(desc(dAIC)) %>% slice_head(n = K)
  }
  
  #terms = top_k_predictors(mod)
  
  effect_vectors <- function(fit, df, terms_tbl,
                             y_cols = c("stock_a","stock_i","stock_s"),
                             cont_step = "sd"){  # "sd" or "q10to90"
    base <- typical_row(df)
    
    # ensure base has all needed columns
    miss <- setdiff(names(df), names(base))
    if(length(miss)) base[miss] <- df[1, miss]
    
    # baseline prediction
    mu0 <- as.data.frame(predict(fit, newdata = base, type = "response"))
    print(mu0)
    colnames(mu0) <- y_cols
    p0 <- as.numeric(mu0[1, ])
    
    rows <- list()
    for (term in terms_tbl$term){
      newd <- base
      x <- df[[term]]
      if(is.numeric(x)){
        ux <- sort(unique(x)); ux <- ux[!is.na(ux)]
        if(length(ux) == 2 && all(ux %in% c(0,1))){  # binary (e.g., extinction, low pop) -> flip to 1
          newd[[term]] <- 1
        } else { # continuous (e.g., log popsize)
          if(cont_step == "sd"){
            newd[[term]] <- as.numeric(base[[term]]) + sd(x, na.rm=TRUE)
          } else { # move from 10% to 90% value
            q <- quantile(x, c(.1,.9), na.rm=TRUE)
            newd[[term]] <- q[2]
          }
        }
      } else {
        # factors/characters: set to most frequent level (already is), skip
        next
      }
      mu1 <- as.data.frame(predict(fit, newdata = newd, type = "response"))
      colnames(mu1) <- y_cols
      p1 <- as.numeric(mu1[1, ])
      
      rows[[term]] <- data.frame(
        term = term,
        x0 = p0[1], y0 = p0[2], z0 = p0[3],
        x1 = p1[1], y1 = p1[2], z1 = p1[3],
        dx = p1[1]-p0[1], dy = p1[2]-p0[2], dz = p1[3]-p0[3],
        length = sqrt((p1[1]-p0[1])^2 + (p1[2]-p0[2])^2 + (p1[3]-p0[3])^2)
      )
    }
    bind_rows(rows)
  }
  
  
  plot_effect_ternary <- function(fit, df, K = 5, y_cols = c("stock_a","stock_i","stock_s"),
                                  cont_step = "sd"){
    top_tbl <- top_k_predictors(fit, df, K)  # choose by ΔAIC (mean side)
    vecs <- effect_vectors(fit, df, top_tbl, y_cols, cont_step)
    
    # scale arrow sizes for visibility
    smax <- max(vecs$length)
    vecs$lw <- 0.5 + 2.5 * vecs$length / smax
    
    base_point <- vecs[1, c("x0","y0","z0")]  # same for all terms
    base_point$label <- "Baseline"
    
    p <- ggtern() +
      # baseline point
      geom_point(data = base_point, aes(x = x0, y = y0, z = z0), size = 3, color = "black") +
      # effect arrows
      geom_segment(data = vecs,
                   aes(x = x0, y = y0, z = z0, xend = x1, yend = y1, zend = z1),
                   arrow = arrow(length = unit(0.2, "cm")), color = "firebrick",linewidth=0.4) +
      scale_size_continuous(range = c(0.6, 2.5), guide = "none") +
      labs(title = "Stocking distribution: effect vectors of top predictors",
           T = y_cols[2], L = y_cols[1], R = y_cols[3],
           subtitle = "Arrows show shift in predicted composition from baseline when each predictor increases (bin: 0→1, cont: +1 SD)") +
      theme_showarrows() + theme_bw()
    
    output = list()
    output$p = p
    output$vecs = vecs
    output$top_tbl = top_tbl
    return(output)
  }
}
out <- plot_effect_ternary(mod11.2, falldf_analysis, K = 5,
                           y_cols = c("stock_a","stock_i","stock_s"),
                           cont_step = "sd")
out$vecs # origin point of the vectors.
out$top_tbl # vector coordinates
out[[1]] # plot
# save vecs as csv
modname= 'mod11.2'
ofilename = sprintf('G:/My Drive/research/nmsu/hatchery operation/codes/dynamic programming2/manuscript_results/stockingdistribution_effectvectors_seed%d_paramset%d_c%d_%s_hatchery3_3_7.csv',params[1],params[2],params[3],modname)
write.csv(out$vecs,ofilename,row.names=FALSE)




