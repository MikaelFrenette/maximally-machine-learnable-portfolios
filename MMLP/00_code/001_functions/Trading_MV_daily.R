Trading_MV_daily <- function(backpack_data,backpack_trading,h=1){
  
  # --- This function calculates the trading-positions of your portfolios that a mean-variance optimizing investor would take
  
  
  # --- Part I:     Portfolios that you specify in 'backpack_trading['my_portfolios_name']', e.g.:
  #         1. MACE:          Trading-Positions are taken based on predictions of a Random-Forest
  
  # --- Part II:    The following portfolios will ALWAYS be computed:
  #         2. 'PM'-versions of 'Part I':     Trading-Positions are taken based on a rolling-mean of all portfolios specified in backpack_trading['my_portfolios_name']'
  
  
  
  # ------------------------------------ Couple of Checks up front ---------------------------------------- #
  
  test <- unique(length(backpack_trading[['my_portfolios_name']]),
                 length(backpack_trading[['my_portfolios_weights_raw']]),
                 length(backpack_trading[['my_portfolios_predictions_OOS']]))
  
  if (length(test) > 1){
    stop('Number of Portfolios != Number of Sets of corresponding Weights')
  } 
  
  # ---------------------------------------------------------------------------------------------- #
  
  
  
  # --- How many portfolios are user-specified?
  N_portfolios <- length(backpack_trading[['my_portfolios_name']])
  
  # --- Trade the following portfolios:
  portfolio_names <- c(backpack_trading[['my_portfolios_name']],
                       paste0(backpack_trading[['my_portfolios_name']],' (PM)'))
  
  # --- Unpack the data
  X_train <- backpack_data[['X_train']]
  X_test <- backpack_data[['X_test']]
  
  Y_train <- as.matrix(backpack_data[['Y_train']])
  Y_test <- as.matrix(backpack_data[['Y_test']])
  
  
  # --- Get the time-dimensions of the respective sets:
  TT_train <- dim(Y_train)[1]
  TT_test <- dim(Y_test)[1]
  
  
  # --- Storage for Portfolio Returns: Out-Of-Sample -- Trading following Mean-Variance optimization
  df_MV_R_Blist <- setNames(rep(list(data.frame(matrix(NA,
                                                       nrow=TT_test,
                                                       ncol=backpack_trading[['B']],
                                                       dimnames=list(c(),paste0("B",c(1:backpack_trading[['B']])))))),
                                times=length(backpack_trading[['my_portfolios_name']])*2), 
                            c(backpack_trading[['my_portfolios_name']],
                              paste0(backpack_trading[['my_portfolios_name']], ' (PM)'))) 
  
  
  # --- Storage for Portfolio Returns: Out-Of-Sample -- Trading following Mean-Variance optimization
  df_MV_R <- data.frame(matrix(NA,nrow=TT_test,
                               ncol=length(portfolio_names)+1,
                               dimnames=list(c(),c('time',portfolio_names))),check.names = F)
  df_MV_R['time'] <- backpack_data$time[(length(backpack_data$time)-nrow(backpack_data$Y_test)+1):length(backpack_data$time)]
  
  # --- Collect the parameters for your Mean-Variance Optimization: risk-tolerance, leverage, etc.
  params_MV <- list("lookback"=backpack_trading[['MV_lookback']],
                    "gamma"=backpack_trading[['MV_gamma']],
                    "alpha"=backpack_trading[['MV_alpha']],
                    "MV_pos_min"=backpack_trading[['MV_pos_min']],
                    "MV_pos_max"=backpack_trading[['MV_pos_max']],
                    "vol_model"=backpack_trading[['vol_model']],
                    "H"=h)
  
  
  
  # ============================================================================================== #
  #
  #                  Trading:   Portfolios in backpack_trading['my_portfolios_name']
  #
  # ============================================================================================== #
  
  
  for (nn in 1:N_portfolios){
    
    
    # --- Get the raw weights for portfolio 'nn'
    weights_nn <- backpack_trading[['my_portfolios_weights']][[nn]]
    
    
    for (bags in 1:backpack_trading[['B']]){
      
      #cat('\nBag: ',bags,'\n')
      
      # --- Get the weights for portfolio 'nn' for bag 'bags'
      weights_nn_b <- as.numeric(weights_nn[bags,])
      
      
      
      # -------------------------- Portfolio: In-Sample Realized Returns  -------------------------------- #
      return_ins <- Y_train %*% weights_nn_b
      
      
      
      
      
      
      # -------------------------- Portfolio: Out-of-Sample Buy & Hold - Realized Returns -------------------------------- #
      return_oos_BH <- Y_test %*% weights_nn_b
      
      
      
      
      # -------------------------- Portfolio: Out-of-Sample Predictions  -------------------------------- #
      return_oos_pred <- backpack_trading[['my_portfolios_predictions_OOS']][[nn]][[bags]] 
      
      # -------------------------- Portfolio: Out-of-Sample Predictions (Prevailing Mean) -------------------------------- #
      return_oos_pred_PM <- roll_mean(series_oos=return_oos_BH,
                                      series_ins=return_ins,
                                      lookback=backpack_trading[['MV_lookback']],h)
      
      
      # -------------------------- Mean-Variance Exercise  -------------------------------- #
      # --- Portfolio
      MV_out <- MeanVariance(return_oos_BH,return_oos_pred,return_ins,params_MV)
      MV_return <- MV_out$return
      # --- Portfolio (Prevailing Mean)
      MV_out_PM <- MeanVariance(return_oos_BH,return_oos_pred_PM,return_ins,params_MV)
      MV_return_PM <- MV_out_PM$return
      
      
      # --------------------- Store the Out-of-Sample Returns for the current 'bag' ------------------------ #
      
      df_MV_R_Blist[[backpack_trading[['my_portfolios_name']][nn]]][,paste0('B',bags)] <- MV_return
      df_MV_R_Blist[[paste0(backpack_trading[['my_portfolios_name']][nn],' (PM)')]][,paste0('B',bags)] <- MV_return_PM
      
    }
    
    
    
    # -------------------------- Store the Out-of-Sample Returns  -------------------------------- #
    df_MV_R[backpack_trading[['my_portfolios_name']][nn]]  <- apply(df_MV_R_Blist[[backpack_trading[['my_portfolios_name']][nn]]],1,mean)
    df_MV_R[paste0(backpack_trading[['my_portfolios_name']][nn],' (PM)')] <- apply(df_MV_R_Blist[[paste0(backpack_trading[['my_portfolios_name']][nn],' (PM)')]],1,mean)
    
  }
  
  
  
  out <- list('MV_returns'=df_MV_R,'MV_out'=MV_out)
  
  
  return(out)
  
  
}