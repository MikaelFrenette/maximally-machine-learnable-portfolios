#################################################################################################################
#
#
#                                   MAXIMALLY MACHINE LEARNABLE PORTFOLIOS
#                             Goulet Coulombe, Philippe and Göbel Maximilian (2023)
#
#                 1.    MMLP - Estimation
#                 2.    Trading
#                 3.    Post-Trading Evaluation
#                 4.    Plotting of Cumulative Returns
#
#
#################################################################################################################
rm(list = ls())


library(lubridate)
library(dplyr)
library(ranger)
library(glmnet)
library(RColorBrewer)
library(foreach)
library(parallel)
library(zoo)

load_pckgs <- c('lubridate','dplyr','ranger','glmnet','foreach','parallel','zoo')

set.seed(1234)

backpack_hyps <- list()
backpack_trading <- list()
backpack_data <- list()

# =================================== USER INTERACTION ========================================= #


# ------------------------- Set your directory to the main folder -------------------- #
directory <- '[YOUR DIRECTORY TO THE MAIN FOLDER]'


backpack_trading['directory'] <- directory
backpack_data['directory'] <- directory
backpack_data[['load_pckgs']] <- load_pckgs

# -------------------------- Parameters: Data ------------------------------------- #

# --- Which DataSet do you want to load? ['MACE_paper']
backpack_data['load__dataset'] <- 'MACE_paper'


# --- How many stocks do you want to include?
# --- --- ---> paper: 20; 50; 100
backpack_data['N_stocks'] <- 100

# --- Specify the number of feature-lags: [c(start:end)] 
# --- --- ---> paper: c(1:21)
backpack_data[['lags']] <- c(1:21)


# --- For the 'daily' application: at which date do you want your training-sample to end?
# --- --- ---> paper: '2016-12-31'
backpack_data['train_end'] <- '2016-12-31'


# --- Want to build additional features?
# --- NULL: no additional features
# --- --- 'lookback' = c(NULL) ---> historical volatility is used (burn-in: 251 observations)
# --- --- 'estimator' = c('exp__1) ---> no exponential weighting is applied
# ATTENTION: all estimates already come as Lag-1 ! Hence, the estimate at 't' is based on information up until time 't-1'
# --- --- ---> paper: NULL
backpack_data[['build_features']] <- list('vol'=list('lags'=c(1:5),
                                                     'lookback'=c(5,21,63,252),
                                                     'estimator'=c('exp__0.94','exp__1')))
backpack_data[['build_features']] <- list('vol'=list('lags'=c(1:5),
                                                     'lookback'=c(5,21,63,252),
                                                     'estimator'=c('exp__0.94','exp__0.99')))

backpack_data[['build_features']] <- c(NULL)



# --- Want to include exogenous signals? [TRUE; FALSE]
# --- --- ---> paper: not implemented, i.e.: FALSE
backpack_data[['get_exog']] <- F


# -------------------------- Parameters: Bagging ------------------------------------- #

# --- How many Bags do you want to pack? 
# --- ---> If you do not wish to run a bagging strategy, set B=1
# --- --- ---> paper: 1; 50: for plain bagging; 100: for loose bagging
backpack_hyps['B'] <- 1



# -------------------------- Parameters: MACE ------------------------------------- #


# --- Set the Block-Size (an integer for the number of observations to be sampled in blocks) for the 'block.sampler':
# --- --- ---> paper: 42
backpack_hyps['my_blocksize'] <- 42

# --- Global Hyperparameters
# --- --- Do you want to tune "lambda" via cross-validation?
# --- --- ---> paper: T
backpack_hyps['I_want_cv'] <- T
# --- --- How many folds do you want?
# --- --- ---> paper: 10
backpack_hyps['N_cv_folds'] <- 10
# --- --- How many iterations?
# --- --- ---> paper:  N_stocks==20: 250; N_stocks==50: 250; N_stocks==100: 500; 
backpack_hyps['maxit'] <- 250*I(backpack_data[['N_stocks']]==20) + 250*I(backpack_data[['N_stocks']]==50) + 500*I(backpack_data[['N_stocks']]==100)


# --- --- Outer-Learning-Rate:
# --- --- ---> paper:  N_stocks==20: 0.01; N_stocks==50: 0.01; N_stocks==100: 0.05; 
backpack_hyps['lr'] <- 0.01*I(backpack_data[['N_stocks']]==20) + 0.01*I(backpack_data[['N_stocks']]==50) + 0.05*I(backpack_data[['N_stocks']]==100)



# --- RHS: Initialization
# --- --- Estimate variance-covariance matrix on a random subsample of training observations?
# --- --- --- insert any number: ]0,1]
# --- --- --- if you want all observations, insert: 0
# --- --- ---> paper (plain MACE):  0
# --- --- ---> paper (bag MACE): 0.8
backpack_hyps['rhs_init_cov_sample'] <- 0
# --- --- Variance-Covariance Shrinkage Estimator, in case Sample-Covariance is singular:
# --- --- ---> 'LW03' = Ledoit & Wolf (2003); 'LW04' = Ledoit & Wolf (2004);
backpack_hyps[['rhs_init_cov_sample_SHRINKAGE']] <- 'LW03'

# --- RHS: Random Forest
# --- --- How many trees do you wanna plant?
# --- --- ---> paper:  1500
backpack_hyps['my_trees'] <- 1500
backpack_trading['my_trees'] <- backpack_hyps['my_trees']
# --- --- Set the min.node.size:
# --- --- ---> paper:  200
backpack_hyps['my_minnodesize'] <- 200

# --- --- 'mtry' will be: ncol(X)/mtry_denom ---> Set 'mtry_denom':
# --- --- ---> paper:  10
backpack_hyps['mtry_denom'] <- 10

# --- --- Minimum number of observations in terminal node:
# --- --- ---> paper: 1
backpack_hyps['min_bucket'] <- 1



# --- LHS: glmnet
# --- --- Type of Regularization? [0 = Ridge; 1 = LASSO]
# --- --- ---> paper:  0.0
backpack_hyps['my_alpha'] <- 0.0
# --- --- Set "lambda" manually. (Overwritten if I_want_cv == TRUE or I_want_lambda_targeting == TRUE)
# --- --- paper: no specific default value, as either 'I_want_cv== TRUE' or 'I_want_lambda_targeting == TRUE'
backpack_hyps['my_lambda'] <-  0.02^2*backpack_hyps[['my_blocksize']]
# --- --- Lower-bound on glmnet-betas? [-Inf;Inf]
# --- --- ---> paper:  -3
backpack_hyps['my_lowerbound'] <- -3
# --- --- Upper-bound on weights? [-Inf;Inf; NULL == no bound]
# --- --- ---> paper:  NULL
backpack_hyps['my_upperbound'] <- NULL

# --- --- Do you want to impose a stock-specific penalty (SD of a given stock-return series)?
# --- --- ---> paper:  T
backpack_hyps['I_want_stockspecificpenalty'] <- T
# --- --- --- Do you want to safeguard against any market-frenzy?
# --- --- ---> paper:  F
backpack_hyps['I_want_penalty_frenzy'] <- F
# --- --- Do you want to weight individual observations?
# --- --- ---> paper (plain MACE):  F
# --- --- ---> paper (bag/loose bag MACE):  T
backpack_hyps['I_want_obsweight'] <- F
# --- Do you want an intercept?
# --- --- ---> paper (plain MACE):  T
# --- --- ---> paper (min-ret-constrained MACE):  F
backpack_hyps['I_want_intercept'] <- T
# --- Do you want to set some unconditional portfolio-return different from 0? ---> e.g. the daily equivalent of annualized 10%: 0.1/250
# --- --- ---> paper (plain MACE):  0
# --- --- ---> paper (min-ret-constrained MACE):  1
backpack_hyps['c0'] <- 0
if (!(backpack_hyps['c0'] == 0)){
  backpack_hyps['I_want_intercept'] <- F
}

# --- --- LHS: lambda-targeting
# --- --- Do you want Lambda-Targeting?
# --- --- ---> paper:  T
backpack_hyps['I_want_lambda_targeting'] <- T
# --- --- --- How do you want to target lambda? By a pre-defined R2? By the highest 'dev.ratio'? ['R2','dev_ratio']
# --- --- ---> paper:  'R2'
backpack_hyps['my_lambda_target'] <- 'R2'
# --- --- --- In case you chose: 'my_lambda_target==R2', what's the R2 that you are targeting? [admissible values: 0,...,1]
# --- --- ---> paper (plain/bag MACE):  0.01
# --- --- ---> paper (loose-bag MACE):  0.02
backpack_hyps['my_R2_target'] <- 0.01


# --- --- LHS -- prediction: lambda-tranquilizer
# --- --- There is the possibility to modify the chosen lambda further: 
# --- --- --- upweight: ] 0, 1 [
# --- --- --- downweight: ] 1, Inf [
# --- --- ---> paper:  1
backpack_hyps['lambda_tranquilizer'] <- 1


# --- Do you want Early-Stopping? 
#     ---> Based on RMSE of Validation Set 
#     ---> If there is NO Validation Set: based on RMSE of Training Set
# --- --- ---> paper:  F
backpack_hyps['I_want_ES'] <- FALSE 

# --- --- What is your patience-period for ES to get triggered?
# --- --- ---> paper:  None
backpack_hyps['ES_patience'] <- 100



# -------------------------- Parameters: Portfolio-Evaluation ---------------------- #

# --- Type of Volatility-Modeling: ['ewma','albaMA]
# --- ---> paper: 'ewma'
backpack_hyps['vol_model'] <- 'ewma'

# --- Define the minimum and maximum position you are willing to trade your portfolio your portfolio
backpack_trading['MV_pos_max'] <- 2
backpack_trading['MV_pos_min'] <- -1

# --- What is the lookback-period for calculating the prevailing mean?
# --- ---> paper: 2520
backpack_trading['MV_lookback'] <- 120*21


# --- for the Mean-Variance-Portfolio Exercise:
# --- --- 'Gamma' = Coefficient of Relative Risk Aversion 
# --- --- 'Alpha' = Decay-Parameter for computing the Exponentially-Weighted Moving-Average of the Portfolio-Variance
backpack_trading['MV_gamma'] <- 5
backpack_trading['MV_alpha'] <- 0.94


# ============================================================================================== #


# --- Halflife:
#halflife <- log(0.5)/log(backpack_trading[['MV_alpha']])


# ============================================================================================== #
#
#                             00.   Load the Data & Auxiliary Functions
#
# ============================================================================================== #

# --- Load MACE
source(file.path(directory,"00_code/001_functions/MACE_daily.R"))

# --- Load the function for Trading
source(file.path(directory,"00_code/001_functions/Trading_MV_daily.R"))

# --- Load some helper functions
source(file.path(directory,"00_code/001_functions/auxiliaries.R"))




# --- Load the data: 
data_raw <- read.csv(file.path(directory,paste0('10_data/daily__',backpack_data[['load__dataset']],'.csv')))[,c(1:(backpack_data[['N_stocks']] + 1))]


# --- Load exogenous features, if requested:
if (backpack_data[['get_exog']]){
  data_exog <- read.csv(file.path(directory,'10_data/daily__MFS_zScores.csv'), row.names = 1)
  
  # --- Reduce to the time-domain of 'data_raw'
  data_exog <- data_exog[which(rownames(data_exog) %in% data_raw[,'date']),]
  
  # --- Prepare Training- & Test-Set
  data_exogTrain <- data_exog[data_raw['date'] <= backpack_data[['train_end']],]
  data_exogTest <- data_exog[data_raw['date'] > backpack_data[['train_end']],]
  
  
} else {
  data_exogTrain <- data.frame()
  data_exogTest <- data.frame()
}

# -------------------------------------- Generate Training- and Test-Split ----------------------------------------------- #

data <- list('X'=data_exogTrain, 
             'Xtest'=data_exogTest,
             'Y'=data_raw[data_raw['date'] <= backpack_data[['train_end']],][-1],
             'Ytest'=data_raw[data_raw['date'] > backpack_data[['train_end']],][-1],
             'time'=data_raw['date']
             )


# -------------------------------------- Pack the Data ----------------------------------------------- #


# --- Feature-Set: Training-Data  (format:  data.frame)
backpack_data[['X_train']] = data[['X']]

# --- Feature-Set: Test-Data  (format: data.frame)
backpack_data[['X_test']] = data[['Xtest']]

# --- Labels: Training-Data  (format:  matrix)
backpack_data[['Y_train']] = data[['Y']]

# --- Feature-Set: Test-Data  (format: matrix)
backpack_data[['Y_test']] = data[['Ytest']]

# --- Dates:
backpack_data[['time']] = data[['time']][['date']]






#############################################################################################################
#
#                              1.       MMLP - Estimation
#
############################################################################################################

# -------------------------------------- MMLP ----------------------------------------------- #

MACE_out <- MACE_daily(backpack_data,backpack_hyps,verbose=T)



############################################################################################################
#
#                              2.       Trading
#
############################################################################################################


# =================================== USER INTERACTION ========================================= #


# --- Insert the name of the portfolios you want to trade:
backpack_trading['my_portfolios_name'] <- c('MACE')


# --- Insert the raw weights for each portfolio that you want to trade:
backpack_trading[['my_portfolios_weights']] <- list('MACE' = MACE_out$df_MACE_weights_Bdf)


# --- Insert the Out-Of-Sample predictions for your portfolio
backpack_trading[['my_portfolios_predictions_OOS']] <- list('MACE' = MACE_out$df_pred_Blist$Test$MACE)


# =================================== USER INTERACTION ========================================= #





# ----------------------- Copy some of the parameters that went into MACE ---------------------- #

backpack_trading['B'] <- backpack_hyps['B']
backpack_trading['my_blocksize'] <- backpack_hyps['my_blocksize']
backpack_trading['my_minnodesize'] <- backpack_hyps['my_minnodesize']
backpack_trading['mtry_denom'] <- backpack_hyps['mtry_denom']
backpack_trading['market_neutral'] <- backpack_hyps['market_neutral']
backpack_trading['vol_model'] <- backpack_hyps['vol_model']

# ---------------------------------------------------------------------------------------------- #





# -------------------------------------- Trading ----------------------------------------------- #

Trading_out <- Trading_MV_daily(backpack_data,backpack_trading,h=1)




############################################################################################################
#
#                              3.       Post-Trading Evaluation
#
############################################################################################################



my_freq <- 250

# --- Annualized Average Return
print(paste0('Annualized Return -- MACE:         ', round(returnA(Trading_out$MV_returns[['MACE']],my_freq)*100,1), '%'))
print(paste0('Annualized Return -- MACE (PM):    ', round(returnA(Trading_out$MV_returns[['MACE (PM)']],my_freq)*100,1), '%'))

# --- Sharpe Ratio
print(paste0('Sharpe Ratio -- MACE:         ', round(sharpe_ratio(Trading_out$MV_returns[['MACE']],my_freq),3)))
print(paste0('Sharpe Ratio -- MACE (PM):    ', round(sharpe_ratio(Trading_out$MV_returns[['MACE (PM)']],my_freq),3)))


# --- Gross Exposure:
print(paste0('Gross Exposure -- MACE:         ', round(sum(abs(MACE_out$df_MACE_weights_Bdf)),3)))


# --- R2
print(paste0("R2 (Training-Set):         ", round(mean(MACE_out$R2_train_Bvec),3)))
print(paste0("R2 (Test-Set):             ", round(mean(MACE_out$R2_test_Bvec),3)))




############################################################################################################
#
#                              4.       Plotting:     Cumulative Returns
#
############################################################################################################



library(ggplot2)
scaleFUN <- function(x) sprintf("%.2f", x)


# ----------------------------- Get U.S. Recession Dates on a Monthly Frequency ----------------------------- #
NBER_Recessions_dates <- get_Recessions(region='US',freq=12, start_year=1960, end_year=2024)


# ----------------------------- Collect the Cumulative Returns ----------------------------- #
plot_df <- data.frame("time"=as.Date(Trading_out$MV_returns[['time']], format="%Y-%m-%d"), 
                      "MACE"=cumsum(Trading_out$MV_returns[['MACE']]),
                      "MACEpm"=cumsum(Trading_out$MV_returns[['MACE (PM)']]))




# ----------------------------- Go Plotting! ----------------------------- #

my_graph <- ggplot() +
  geom_line(data = plot_df,aes(x =time , y =MACEpm, color="MACE (PM)"), linewidth = 1.6) +
  geom_line(data = plot_df,aes(x =time , y =MACE, color="MACE"), linewidth = 1.6) + 
  geom_hline(yintercept=0, color="black", linewidth = 0.75) +
  theme_bw()+
  xlab('')+ylab('')+labs(color='')+
  theme(axis.text.x=element_text(size=24,angle = 0,hjust=1),
        axis.text.y=element_text(size=24),
        strip.text = element_text(face="bold", size=30),
        legend.text=element_text(size=30),
        legend.position='bottom') +
  scale_color_manual(breaks=c("MACE","MACE (PM)"),
                     values=c("MACE"="#ef3b2c",
                              "MACE (PM)" ="orange"),
                     labels=c("MACE","MACE (PM)"))+ 
  scale_y_continuous(breaks = function(x) break_function(x),labels=scaleFUN,
                     expand = c(0.00,0.00))+
  scale_x_date(breaks = scales::pretty_breaks(n = 8),date_labels = "%Y",
               expand = c(0.005,0.005), limits = c(plot_df$time[1],plot_df$time[nrow(plot_df)]))+
  geom_rect(data=NBER_Recessions_dates, aes(xmin=start, xmax=end, ymin=-Inf, ymax=+Inf), fill='pink', alpha=0.3)+
  theme_Publication() +  
  theme(legend.text=element_text(size=14),legend.key.size = unit(3,"line"),
        legend.position = "bottom",
        legend.margin=margin(-10,25,25,-10),
        legend.box.margin=margin(-40,-40,-40,-40),
        strip.text = element_text(face="bold", colour = "white",size=30,family="Helvetica"), #
        strip.background=element_rect(colour="black",fill="black"),
        plot.margin = grid::unit(c(5,11,5,-11), "mm")) + 
  guides(col=guide_legend(nrow=1,byrow=F)) 

my_graph


