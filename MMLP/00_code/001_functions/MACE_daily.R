MACE_daily <- function(backpack_data,backpack_hyps,verbose=FALSE){
  
  set.seed(1234)
  
  # --- Unpack the data
  X_train_init <- backpack_data[['X_train']]
  X_test_init <- backpack_data[['X_test']]
  
  Y_train <- as.matrix(backpack_data[['Y_train']])
  Y_test <- as.matrix(backpack_data[['Y_test']])
  
  
  # --- Get the time-dimensions of the respective sets:
  TT_train <- dim(Y_train)[1]
  TT_test <- dim(Y_test)[1]
  
  
  # --- Some preliminary dataframes:
  rmse_train_Bdf <- data.frame(matrix(NA,nrow=backpack_hyps[['maxit']],ncol=backpack_hyps[['B']],
                                      dimnames = list(c(),paste0("B",c(1:backpack_hyps[['B']])))))
  rsq_train_Bdf <- data.frame(matrix(NA,nrow=backpack_hyps[['maxit']],ncol=backpack_hyps[['B']],
                                     dimnames = list(c(),paste0("B",c(1:backpack_hyps[['B']])))))
  rsq_test_Bdf <- data.frame(matrix(NA,nrow=backpack_hyps[['maxit']],ncol=backpack_hyps[['B']],
                                    dimnames = list(c(),paste0("B",c(1:backpack_hyps[['B']])))))
  hx_train_Bdf <- data.frame(matrix(NA,nrow=TT_train,ncol=backpack_hyps[['B']],
                                    dimnames = list(c(),paste0("B",c(1:backpack_hyps[['B']])))))
  hx_test_Bdf <- data.frame(matrix(NA,nrow=TT_test,ncol=backpack_hyps[['B']],
                                   dimnames = list(c(),paste0("B",c(1:backpack_hyps[['B']])))))
  hy_train_Bdf <- data.frame(matrix(NA,nrow=TT_train,ncol=backpack_hyps[['B']],
                                    dimnames = list(c(),paste0("B",c(1:backpack_hyps[['B']])))))
  hy_test_Bdf <- data.frame(matrix(NA,nrow=TT_test,ncol=backpack_hyps[['B']],
                                   dimnames = list(c(),paste0("B",c(1:backpack_hyps[['B']])))))
  fit_train_Bdf <- data.frame(matrix(NA,nrow=TT_train,ncol=backpack_hyps[['B']],
                                     dimnames = list(c(),paste0("B",c(1:backpack_hyps[['B']])))))
  fit_test_Bdf <- data.frame(matrix(NA,nrow=TT_test,ncol=backpack_hyps[['B']],
                                    dimnames = list(c(),paste0("B",c(1:backpack_hyps[['B']])))))
  
  # --- Vectors to store results
  my_run_Bvec <- rep(NA,times=backpack_hyps[['B']])
  R2_train_Bvec <- rep(NA,times=backpack_hyps[['B']])
  R2_test_Bvec <- rep(NA,times=backpack_hyps[['B']])
  
  # --- Lists to store results
  r1_hist_Blist <- rep(list(NA),times=backpack_hyps[['B']])
  r2_hist_Blist <- rep(list(NA),times=backpack_hyps[['B']])
  
  
  # --- Storage for portfolio-weights
  df_MACE_weights_Bdf <- data.frame(matrix(NA, nrow=backpack_hyps[['B']], 
                                           ncol=dim(Y_train)[2],dimnames=list(c(),colnames(backpack_data[['Y_train']]))))
  
  
  
  df_MACE_weights_raw_Bdf <- data.frame(matrix(NA, nrow=backpack_hyps[['B']], 
                                                ncol=dim(Y_train)[2]+1,
                                               dimnames=list(c(),c('c',colnames(backpack_data[['Y_train']])))))

  
  # --- Storage for Predictions
  df_pred_Blist <- list('Train'=setNames(rep(list(data.frame(matrix(NA,
                                                                    nrow=TT_train,
                                                                    ncol=backpack_hyps[['B']],
                                                                    dimnames=list(c(),paste0("B",c(1:backpack_hyps[['B']])))))),
                                             times=6), c('MACE','MACE (PM)','EW (RF)', 'EW (PM)', 'SP500 (RF)', 'SP500 (PM)')),
                        'Test'=setNames(rep(list(data.frame(matrix(NA,
                                                                   nrow=TT_test,
                                                                   ncol=backpack_hyps[['B']],
                                                                   dimnames=list(c(),paste0("B",c(1:backpack_hyps[['B']])))))),
                                            times=6), c('MACE','MACE (PM)','EW (RF)', 'EW (PM)', 'SP500 (RF)', 'SP500 (PM)')))
  
  
  
  
  ################################################################################################
  #
  #                    1)       Run MACE (Algorithm 1 in  Goulet Coulombe & Goebel (2023))
  #
  ################################################################################################
  
  # --- Create and register the Parallel Clusters
  #n_cores <- parallel::detectCores()
  #my_cluster <- parallel::makeCluster(min(backpack_hyps[['B']],n_cores))
  #doParallel::registerDoParallel(cl = my_cluster)
  
  #list__B_out <- foreach(bags = c(1:backpack_hyps[['B']]), .packages = backpack_data[['load_pckgs']]) %dopar% {
    
    
    # --- Load some helper functions
    #source(file.path(backpack_data[['directory']],"00_code/000_MACE/001_functions/auxiliaries.R"))

  list__B_out <- list()
  for (bags in 1:backpack_hyps[['B']]){  
    
    if (verbose){
      print(paste0("Running Bag ", bags,"/",backpack_hyps[['B']]))
    }
    
    
    # --- Some preliminary storages:
    rsq_train_vec <- c()
    rsq_test_vec <- c()
    rmse_train_vec <- c()
    cor_train_vec <- c()
    oob_MSE_rhs_vec <- c()
    
    r1_hist <- list() # List to store predictions
    r2_hist <- list() # List to store predictions
    
    # ---------------------------------------------------------------------------------------------------------- #
    #
    #                                     Algorithm 1 - Step 1   
    #                 -   Initialize 'z_0' (denoted: z2) as either 
    #                                 (i)   the scaled inverse-volatility weighted portfolio or 
    #                                 (ii)  the scaled equally-weighted portfolio
    #                 -   Initialize 'f*_s' (denoted: z1) as a draw from N(0,1)
    #
    # ---------------------------------------------------------------------------------------------------------- #
    
    # --- Initialize 'z1' 
    z1 = rnorm(nrow(Y_train))
    
    # --- Initialize 'z2' & portfolio-weights
    MACE_betas <- matrix(NA, backpack_hyps[['maxit']]+1, 1+ncol(Y_train))
    
    
    
    if (backpack_hyps[['my_lowerbound']] < 0){
      
      # --- Select a certain sub-set of observations for calculating the variance-covariance matrix
      if (backpack_hyps[['rhs_init_cov_sample']] == 0){
        idx <- c(1:nrow(Y_train))
      } else {
        bs=block.sampler(Y_train,  sampling_rate=backpack_hyps[['rhs_init_cov_sample']], 
                         block_size = backpack_hyps[['my_blocksize']],num.tree=backpack_hyps[['B']])
        idx <- c(1:nrow(Y_train))[as.logical(bs$inbag1[[bags]])]
      }
      
      # --- Inverse covariance weighting:
      # --- --- Is the matrix singular (i.e. not invertible?)
      # --- --- --- Eigenvalues of the Co-Variance Matrix:
      E_vals <- eigen(cov(Y_train[idx,]))$values
      if (any(abs(E_vals) <= 1e-10)){
        # --- System is singular: deploy the covariance-shrinker
        if (backpack_hyps[['rhs_init_cov_sample_SHRINKAGE']] == 'LW03'){
          cov_Y <- do__CovShrinkage_LW03(Y_train[idx,])$shrinkage_COV
        } else if (backpack_hyps[['rhs_init_cov_sample_SHRINKAGE']] == 'LW04'){
          cov_Y <- do__CovShrinkage_LW04(Y_train[idx,])$shrinkage_COV
        }
        inv_cov_weight <- as.vector(solve(cov_Y)%*%rep(1,ncol(Y_train[idx,])))
      } else {
        # --- System is not singular
        inv_cov_weight <- as.vector(solve(cov(Y_train[idx,]))%*%rep(1,ncol(Y_train[idx,])))
      }
      initial_scaler <- 1 / sum(inv_cov_weight) * inv_cov_weight

      z2 = scale(Y_train %*% initial_scaler)
      MACE_betas[1,] <- c(-mean(Y_train %*% initial_scaler), initial_scaler / sd(Y_train %*% initial_scaler))
      
    } else {
      z2 = scale(apply(Y_train,1,mean))
      MACE_betas[1,] <- c(0,rep(1/ncol(Y_train),times=ncol(Y_train)))
    }
    
    
    
    # --- Just for storage
    z1_df <- data.frame("z1_0"=z1)
    z2_df <- data.frame("z2_0"=z2)
    
    
    # ---------------------------------------------------------------------------------------------------------- #
    #
    #                     Algorithm 1 - Step 2:   Loop until 's_max' (denoted: maxit)
    #
    # ---------------------------------------------------------------------------------------------------------- #
    
    
    for(i in 1:backpack_hyps[['maxit']]){
      
      set.seed(1234*bags+i)
      
      if (verbose){
        print(paste0("Iteration ", i,'/', backpack_hyps[['maxit']] ," --- Bag ", bags,"/",backpack_hyps[['B']]))
      }
      
      # ----------------------------------------------------------------------------------------------------- #
      #
      #                     Algorithm 1 - Step 3:  The Random Forest Step
      #
      # ----------------------------------------------------------------------------------------------------- #
      
      
      # --- Create the MARX features of the portfolios
      mace_marx <- matrix(sapply(backpack_data[['lags']], function(x) lag(z2,x)),nrow=length(z2),ncol=length(backpack_data[['lags']]),
                          dimnames = list(c(),paste0('L_',backpack_data[['lags']])))
      mace_marx[is.na(mace_marx)] <- 0
      mace_marx <- t(apply(mace_marx,1,cumsum))
      
      
        
      # --- Attach it to the features
      if (nrow(X_train_init) == 0){
        X_train <- mace_marx
      } else {
        X_train <- cbind(X_train_init, mace_marx)
      }
      
      
      # --------------------------- Attach NEW Features --------------------------- #
      if (!is.null(backpack_data[['build_features']])){
        
        # --- Instantiate the matrix for the new features
        newFEAT <- matrix(nrow=length(z2),ncol=0)
        
        # --- Run over each estimator:
        for (e in backpack_data[['build_features']][['vol']][['estimator']]){
          
          # --- Extract the 'lmbda' parameter:
          e_lmbda <- as.numeric(strsplit(e,'__')[[1]][2])
          
          # --- Runn over each lookback window:
          for (w in backpack_data[['build_features']][['vol']][['lookback']]){
            
            # --- Calculate the volatility estimator:
            out_vol <- vol_expWEIGHT(z2,lmbda=e_lmbda,window=w)
            
            # --- Run across all lags:
            for (l in backpack_data[['build_features']][['vol']][['lags']]){
              # --- REMEMBER: the estimates in 'out_vol' already come as 1-period lagged, i.e. are forecasts for day 't' based on information up until 't-1'
              newFEAT <- cbind(newFEAT,matrix(c(rep(NA,times=l-1),out_vol[1:(length(out_vol)-l+1)]),
                                              dimnames=list(c(),paste0('L',l,'_vol__LB',w,'_exp',e_lmbda))))
              # --- Next: 'l'
            }
            # --- Next: 'w'
          }
          # --- Next: 'e'
        }
        
        # --- Attach to the existing features:
        X_train <- cbind(X_train, newFEAT)
        
      }
      
      
      
      # --- Fit the RHS to our portfolio
      ddd = as.data.frame(cbind(z=z2,X_train))
      # --- Remove NAs:
      ddd <- na.trim(ddd)
      # --- Collect leading NAs ----> for concatenation later on....
      vec_NA <- rep(NA,dim(X_train)[1]-dim(ddd)[1])
      colnames(ddd)[1]='z'
      
      # --- Draw blocks of observations for Out-Of-Bag computations
      bs=block.sampler(ddd,sampling_rate=0.8, 
                       block_size=backpack_hyps[['my_blocksize']], 
                       num.tree=backpack_hyps[['my_trees']])
      
      
      r1 <- ranger(z ~ ., data = ddd,
                   num.trees=backpack_hyps[['my_trees']],
                   inbag=bs$inbag1,
                   min.node.size = backpack_hyps[['my_minnodesize']],
                   min.bucket = backpack_hyps[['min_bucket']],
                   mtry=round(ncol(X_train)/backpack_hyps[['mtry_denom']]), 
                   #importance = 'impurity',
                   num.threads = 2)
      
      
      # ----------------------------------------------------------------------------------------------------- #
      #
      #                     Algorithm 1 - Step 4:  Update the RHS
      #
      # ----------------------------------------------------------------------------------------------------- #
      
      
      # --- --- Adjust 'z1'
      if(i>1){
        z1 = c(vec_NA,backpack_hyps[['lr']]*r1$predictions) + (1-backpack_hyps[['lr']])*z1
      }else{
        z1 = as.vector((r1$predictions))
        # --- Attach leading NAs:
        z1 <- c(vec_NA,z1)
      }
      
      
      
      # ----------------------------------------------------------------------------------------------------- #
      #
      #                     Algorithm 1 - Step 5:  The Ridge Regression Step
      #
      # ----------------------------------------------------------------------------------------------------- #
      
      # --- Create the folds for cross-validation
      N_obs <- nrow(Y_train) - length(vec_NA)
      cv_folds <- as.vector(sapply(c(1:(backpack_hyps[['N_cv_folds']]-1)), function(x) rep(x,N_obs/backpack_hyps[['N_cv_folds']])))
      cv_folds <- c(cv_folds, rep(backpack_hyps[['N_cv_folds']],N_obs-length(cv_folds)))
      
      # --- Fit the LHS to 'z1'
      ddd= as.data.frame(cbind(z=z1,Y_train))
      colnames(ddd)[1]='z'
      # --- Remove NAs:
      ddd <- na.trim(ddd)
      # --- --- Stock-specific Penalty?
      if (backpack_hyps[['I_want_stockspecificpenalty']]){
        ssp <- as.vector(apply(Y_train,2,sd))
      } else {
        ssp <- rep(1, times=ncol(as.matrix(Y_train)))
      }
      # --- --- Observation weights?
      if (backpack_hyps[['I_want_obsweight']]){
        #obsw <- (rexp(nrow(Y_train)))^(1/sqrt(i))
        obsw <- (rexp(N_obs))^(1/i)
        if(i>backpack_hyps[['maxit']]/3){obsw <- rep(1,N_obs)}
      } else {
        obsw <- rep(1,N_obs)
      }
      
      # --- Generate the indices for observations included in the estimation:
      idx_Y <- (length(vec_NA)+1):nrow(Y_train)
      idx_z <- (length(vec_NA)+1):length(z1)
      
      
      if (backpack_hyps[['I_want_lambda_targeting']]) {
        
        
        if (backpack_hyps[['I_want_cv']]){
          
          r2 = cv.glmnet(as.matrix(Y_train[idx_Y,]), z1[idx_z]+backpack_hyps[['c0']], 
                         intercept = backpack_hyps[['I_want_intercept']], 
                         alpha=backpack_hyps[['my_alpha']],
                         nfolds = backpack_hyps[['N_cv_folds']], 
                         lower.limits = backpack_hyps[['my_lowerbound']],
                         weights=obsw, 
                         penalty.factor = ssp,
                         type.measure = "mse",
                         standardize = FALSE,standardize.response = F,
                         foldid = cv_folds)
          
          all_lambdas <- r2$lambda
          
          if (backpack_hyps[['my_lambda_target']] == 'R2'){
            # --- Choose the lambda that best meets your R2-Target
            r2$lambda <- r2$lambda[order(abs(c(r2$glmnet.fit$dev.ratio-backpack_hyps[['my_R2_target']])))[1]]
          } else if (backpack_hyps[['my_lambda_target']] == 'dev_ratio'){
            # --- Choose the lambda that best meets your R2-Target
            r2$lambda <- r2$lambda[which(r2$glmnet.fit$dev.ratio==max(r2$glmnet.fit$dev.ratio))]
          }
          
          # --- Choose the corresponding betas
          r2$beta <- as.matrix(r2$glmnet.fit$beta[,which(all_lambdas == r2$lambda)])
          # --- Choose the corresponding alpha
          r2$a0 <- r2$glmnet.fit$a0[which(all_lambdas == r2$lambda)]
          
          
        } else {
          
          r2 = glmnet(Y_train[idx_Y,], z1[idx_z]+backpack_hyps[['c0']], 
                      intercept = backpack_hyps[['I_want_intercept']], 
                      alpha=backpack_hyps[['my_alpha']], 
                      lower.limits = backpack_hyps[['my_lowerbound']], 
                      type.measure = "mse",
                      standardize=F,
                      standardize.response=F,
                      penalty.factor = ssp,
                      weights = obsw)
          
          # --- Store all penalties:
          all_lambdas <- r2$lambda
          
          if (backpack_hyps[['my_lambda_target']] == 'R2'){
            # --- Choose the lambda that best meets your R2-Target
            r2$lambda <- r2$lambda[order(abs(c(r2$dev.ratio-backpack_hyps[['my_R2_target']])))[1]]
          } else if (backpack_hyps[['my_lambda_target']] == 'dev_ratio'){
            # --- Choose the lambda that best meets your R2-Target
            r2$lambda <- r2$lambda[which(r2$dev.ratio==max(r2$dev.ratio))]
          }
          
          # --- Choose the corresponding betas
          r2$beta <- as.matrix(r2$beta[,which(all_lambdas == r2$lambda)])
          # --- Choose the corresponding alpha
          r2$a0 <- r2$a0[which(all_lambdas == r2$lambda)]
          
          
        }
        
        
        
        
        
        
      } else {
        
        r2 = glmnet(Y_train, z1+backpack_hyps[['c0']], 
                    intercept = backpack_hyps[['I_want_intercept']], 
                    lambda=backpack_hyps[['my_lambda']], 
                    alpha=backpack_hyps[['my_alpha']], 
                    lower.limits = backpack_hyps[['my_lowerbound']], 
                    type.measure = "mse",
                    standardize=F,
                    standardize.response=F,
                    penalty.factor = ssp,
                    weights = obsw)
        
        
        
      }
      
      
      # --- --- Extract the optimal betas ---> needed for adjusting 'z2' in the next step
      opt_betas <- as.matrix(r2$beta)
      
      
      # ----------------------------------------------------------------------------------------------------- #
      #
      #                     Algorithm 1 - Step 6:  Update the LHS
      #
      # ----------------------------------------------------------------------------------------------------- #
      
      
      
      # --- --- Adjust 'z2'
      if (all(abs(opt_betas) <= 1e-20)){
        #stop("All weights == 0...nothing to predict...")
        return_ins <- 0.001*rnorm(length(z2))/i #rep(0,times=length(z2))
        z2 <- backpack_hyps[['lr']]*return_ins + (1-backpack_hyps[['lr']])*z2
        
        # --- Store the (zero) weights:
        MACE_betas[i+1,] <- MACE_betas[i,] * (1-backpack_hyps[['lr']]) + backpack_hyps[['lr']]*c(r2$a0, rep(0,times=length(opt_betas)))
        
        
      } else {
        
          
        # --- Predict the in-sample portfolio returns:
        return_ins <- predict(r2,newx = Y_train, s=r2$lambda)
        if (sd(return_ins) == 0){
          # --- Maybe the penalty was too strong?
          return_ins <- predict(r2,newx = Y_train, s=r2$lambda/5)
        }
        
        # --- Add a little bit of noise
        return_ins_noisy <- as.vector(scale(return_ins))+0.001*rnorm(length(return_ins))/i
        
        if(cor(apply(Y_train,1,mean), return_ins_noisy) < 0){
          return_ins_noisy <- -return_ins_noisy
        }
        
        z2 <- as.vector(scale(backpack_hyps[['lr']]*return_ins_noisy + (1-backpack_hyps[['lr']])*z2))
          
        # --- Store the weights:
        MACE_betas[i+1,] <- MACE_betas[i,] * (1-backpack_hyps[['lr']]) + backpack_hyps[['lr']]*c(r2$a0 - mean(return_ins),
                                                                                                 opt_betas / sd(return_ins))
        
        
      }
      
      
      
      
      
      
      # ----------------------------------------------------------------------------------------------------- #
      #
      #                     Algorithm 1 - Step 7:  Predictions & Early-Stopping Criteria
      #
      # ----------------------------------------------------------------------------------------------------- #
      
      
      # -------------------------- Predictions: Training-Set  -------------------------- #
      
      # --- --- Y-part
      pred_y_train <- z2[idx_z]
      
      
      # --- --- X-part
      pred_x_train <- r1$predictions
      
      
      # --- Add the RMSE of the current iteration
      rmse_train <- sqrt(sum((pred_y_train - pred_x_train)^2)/length(pred_x_train))
      rmse_train_vec <- c(rmse_train_vec,rmse_train)
      
      # --- Check the In-Sample correlation between LHS and RHS
      if (all(opt_betas <= 1e-20)){
        cor_train_vec <- c(cor_train_vec,NA)
      } else {
        cor_train_vec <- c(cor_train_vec,round(cor(pred_y_train,pred_x_train),3))
      }
      
      
      
      
      # -------------------------- Predictions: Test-Set  -------------------------- #
      pred_y_test <- MACE_betas[i,1] + Y_test %*% MACE_betas[i,-1]
        
      if(cor(apply(Y_train,1,mean),return_ins)<0){
        pred_y_test <- -pred_y_test
      }
      
      
      # --- --- X-part
      
      # --- --- --- Create lags of the prediction
      mace_marx <- matrix(sapply(backpack_data[['lags']], function(x) lag(pred_y_test,x)),
                          nrow=length(pred_y_test),ncol=length(backpack_data[['lags']]),
                          dimnames = list(c(),paste0('L_',backpack_data[['lags']])))
      # --- --- --- Fill the missing with in-sample values
      N_obs <- length(pred_y_test)
      for (ii in backpack_data[['lags']]){
        mace_marx[1:min(ii,N_obs),ii] <- head(tail(pred_y_train, ii),min(ii,N_obs))
      }
      
      # --- --- --- Create the MARX features of the portfolios
      mace_marx <- t(apply(mace_marx,1,cumsum))
      
      
      # --- Attach it to the features
      if (nrow(X_test_init) == 0){
        X_test <- mace_marx
      } else {
        X_test <- cbind(X_test_init, mace_marx)
      }
      
      
      # --------------------------- Attach NEW Features --------------------------- #
      if (!is.null(backpack_data[['build_features']])){
        
        # --- Instantiate the matrix for the new features
        newFEAT <- matrix(nrow=length(pred_y_test),ncol=0)
        
        # --- Run over each estimator:
        for (e in backpack_data[['build_features']][['vol']][['estimator']]){
          
          # --- Extract the 'lmbda' parameter:
          e_lmbda <- as.numeric(strsplit(e,'__')[[1]][2])
          
          # --- Run over each lookback window:
          for (w in backpack_data[['build_features']][['vol']][['lookback']]){
            
            # --- Run across all lags:
            for (l in backpack_data[['build_features']][['vol']][['lags']]){
              
              # --- REMEMBER: the estimates in 'out_vol' already come as 1-period lagged, i.e. are forecasts for day 't' based on information up until 't-1'
              
              # --- --- Pre-Attach In-Sample values, so that we don't have any leading NAs:
              series_test <- c(tail(pred_y_train,w+l-1),pred_y_test)
              
              # --- --- Calculate the volatility estimator:
              out_vol <- vol_expWEIGHT(series_test,lmbda=e_lmbda,window=w)
              
              # --- REMEMBER: the estimates in 'out_vol' already come as 1-period lagged, i.e. are forecasts for day 't' based on information up until 't-1'
              newFEAT <- cbind(newFEAT,matrix(out_vol[(w+1):(length(out_vol)-l+1)],
                                              dimnames=list(c(),paste0('L',l,'_vol__LB',w,'_exp',e_lmbda))))
              # --- Next: 'l'
            }
            # --- Next: 'w'
          }
          # --- Next: 'e'
        }
        
        # --- Attach to the existing features:
        X_test <- cbind(X_test, newFEAT)
        
      }
      
      
      
      
      
      pred_x_test <- predict(r1, data = as.data.frame(cbind(X_test)))$predictions
      
      
      
      
      
      # --- Implement Early-Stopping
      if (backpack_hyps[['I_want_ES']]){
        
        # --- Caution:  Early-Stopping is applied on the Training-Set!
        if (i > backpack_hyps[['ES_patience']]){
          if (any(is.na(rmse_train_vec))){
            print("Training-RMSE is NA! Something's wrong!")
            break
          } else {
            if (all(rmse_train_vec[(i-backpack_hyps[['ES_patience']]+1):i] > rmse_train_vec[i-backpack_hyps[['ES_patience']]])){
              ES_iteartion <- i
              print(paste0("Early-Stopping triggered at Iteration: ", i)) 
              break
            }
          }
        }
        
      }
      
      
      # --- Estimate the Model: Future -> Past
      ols_mod <- lm(LHS ~ RHS, data=data.frame("LHS"=c(pred_y_train),"RHS"=pred_x_train))
      
      # --- Test-R2 Monitor
      pred_test <- predict(ols_mod,newdata=data.frame("LHS"=c(pred_y_test),"RHS"=pred_x_test))
      rsq_test <- R2(pred_y_test,pred_test,mean(pred_y_train))
      
      if (rsq_test %in% c(Inf,-Inf)){
        rsq_test_vec <- c(rsq_test_vec,NA)
      } else {
        rsq_test_vec <- c(rsq_test_vec,rsq_test)
      }
      
      # --- Train-R2
      rsq_train_vec <- c(rsq_train_vec,summary(ols_mod)$r.squared)
      
      
      
      # ----------------------------------------------------------------------------------------------------- #
      #
      #                     Store the Out-Of-Bag prediction error
      #
      # ----------------------------------------------------------------------------------------------------- #
      oob_MSE_rhs_vec <- c(oob_MSE_rhs_vec, r1$prediction.error)
      
      
      
      # ----------------------------------------------------------------------------------------------------- #
      #
      #                     Algorithm 1 - Step **:  Collection of Results
      #
      # ----------------------------------------------------------------------------------------------------- #
      
      # --- Store the models
      r1_hist[[length(r1_hist)+1]] <- r1
      r2_hist[[length(r2_hist)+1]] <- r2
      
      # --- Store the Predictable Future and the Past
      z1_df <- cbind(z1_df,z1)
      colnames(z1_df)[ncol(z1_df)] <- paste0("z1_",i)
      z2_df <- cbind(z2_df,z2)
      colnames(z2_df)[ncol(z2_df)] <- paste0("z2_",i)
      
      
      # ----------------------------------------------------------------------------------------------------- #
      #
      #                     Algorithm 1 - Next Iteration
      #
      # ----------------------------------------------------------------------------------------------------- #
      
      
    }
    
    
    ################################################################################################
    #
    #                    2)       Post-Estimation Evaluation
    #
    ################################################################################################
    
    
    print(paste0("Estimation for Bag ", bags," done! Post-Estimation Evaluation."))
    
    # --- Step 1: Pick your run!
    #my_run <- order(as.vector(rollmean(oob_MSE_rhs_vec, k=5, align = c( "center"))))[1]
    my_run <- min(backpack_hyps[['maxit']]-1,which(oob_MSE_rhs_vec==min(oob_MSE_rhs_vec[60:(i-1)]))[1])
    
    # --- Collect the fits
    r1_hist_Blist[[bags]] <- r1_hist
    r2_hist_Blist[[bags]] <- r2_hist
    my_run_Bvec[bags] <- my_run
    
    
    # ====================================================================================== #
    #                           Make Predictions
    # ====================================================================================== #
    
    
    # --- Step 1: Get the MACE-Portfolio-Weights
    MACE_weights <- MACE_betas[my_run,-1] / sum(MACE_betas[my_run,-1])
    
    
    
    
    
    # --- Step 2A: Make Predictions (LHS)
    hy_test <- MACE_betas[my_run,1] + Y_test %*% MACE_betas[my_run,-1]
    hy_train <- MACE_betas[my_run,1] + Y_train[idx_Y,] %*% MACE_betas[my_run,-1]
    
    
    
    
    # --- Step 2B: Make Predictions (RHS)
    
    # --- --- --- Create lags of the prediction
    mace_marx <- matrix(sapply(backpack_data[['lags']], function(x) lag(hy_test,x)),nrow=length(hy_test),ncol=length(backpack_data[['lags']]),
                        dimnames = list(c(),paste0('L_',backpack_data[['lags']])))
    # --- --- --- Fill the missing with in-sample values
    N_obs <- length(hy_test)
    for (ii in backpack_data[['lags']]){
      mace_marx[1:min(ii,N_obs),ii] <- head(tail(hy_train, ii),min(ii,N_obs))
    }
    
    # --- --- --- Create the MARX features of the portfolios
    mace_marx <- t(apply(mace_marx,1,cumsum))
    
    
    # --- --- --- Attach it to the features
    if (nrow(X_test_init) == 0){
      X_test <- mace_marx
    } else {
      X_test <- cbind(X_test_init, mace_marx)
    }
    
    
    # --------------------------- Attach NEW Features --------------------------- #
    if (!is.null(backpack_data[['build_features']])){
      
      # --- Instantiate the matrix for the new features
      newFEAT <- matrix(nrow=length(hy_test),ncol=0)
      
      # --- Run over each estimator:
      for (e in backpack_data[['build_features']][['vol']][['estimator']]){
        
        # --- Extract the 'lmbda' parameter:
        e_lmbda <- as.numeric(strsplit(e,'__')[[1]][2])
        
        # --- Run over each lookback window:
        for (w in backpack_data[['build_features']][['vol']][['lookback']]){
          
          # --- Run across all lags:
          for (l in backpack_data[['build_features']][['vol']][['lags']]){
            
            # --- REMEMBER: the estimates in 'out_vol' already come as 1-period lagged, i.e. are forecasts for day 't' based on information up until 't-1'
            
            # --- --- Pre-Attach In-Sample values, so that we don't have any leading NAs:
            series_test <- c(tail(hy_train,w+l-1),hy_test)
            
            # --- --- Calculate the volatility estimator:
            out_vol <- vol_expWEIGHT(series_test,lmbda=e_lmbda,window=w)
            
            # --- REMEMBER: the estimates in 'out_vol' already come as 1-period lagged, i.e. are forecasts for day 't' based on information up until 't-1'
            newFEAT <- cbind(newFEAT,matrix(out_vol[(w+1):(length(out_vol)-l+1)],
                                            dimnames=list(c(),paste0('L',l,'_vol__LB',w,'_exp',e_lmbda))))
            # --- Next: 'l'
          }
          # --- Next: 'w'
        }
        # --- Next: 'e'
      }
      
      # --- Attach to the existing features:
      X_test <- cbind(X_test, newFEAT)
      
    }
    
    
    
    
    hx_train <- r1_hist[[my_run]]$predictions
    hx_test <- predict(r1_hist[[my_run]], data = as.data.frame(cbind(X_test)))$predictions
    
    
    
    # -------------------------- Get the PREDICTIONS for MACE's OOS-return  -------------------------------- #
    ols_mod <- lm(LHS ~ RHS, data=data.frame("LHS"=hy_train,"RHS"=hx_train))
    
    fit_train <- ols_mod$fitted.values
    fit_test <- predict(ols_mod, newdata=data.frame("LHS"=hy_test,"RHS"=hx_test))
    
    MACE_pred_train <- (fit_train-0*MACE_betas[my_run,1])/sum(as.vector(MACE_betas[my_run,-1]))
    MACE_pred_test <- (fit_test-0*MACE_betas[my_run,1])/sum(as.vector(MACE_betas[my_run,-1]))
    
    
    
    
    # --- Collect the Predictions
    hx_train_Bdf[,paste0("B",bags)] <- c(vec_NA,hx_train)
    hx_test_Bdf[,paste0("B",bags)] <- hx_test
    hy_train_Bdf[,paste0("B",bags)] <- c(vec_NA,hy_train)
    hy_test_Bdf[,paste0("B",bags)] <- hy_test
    fit_train_Bdf[,paste0("B",bags)] <- c(vec_NA,fit_train)
    fit_test_Bdf[,paste0("B",bags)] <- fit_test
    
    # --- Collect the R2
    R2_train_Bvec[bags] <- R2(hy_train,fit_train,mean(hy_train))
    R2_test_Bvec[bags] <- R2(hy_test,fit_test,mean(hy_train))
    
    
    print(paste0("Bag: ", bags, " --- R2 (Training-Set):         ", round(R2_train_Bvec[bags],3)))
    print(paste0("Bag: ", bags, " --- R2 (Test-Set):             ", round(R2_test_Bvec[bags],3)))
    
    
    ################################################################################################
    #
    #                    3)       Store the Bag's Output
    #
    ################################################################################################
    
    
    # --- Some metrics:
    rmse_train_Bdf[,bags] <- c(rmse_train_vec, rep(NA,times=backpack_hyps[['maxit']]-length(rmse_train_vec)))
    rsq_train_Bdf[,bags] <- c(rsq_train_vec, rep(NA,times=backpack_hyps[['maxit']]-length(rsq_train_vec)))
    rsq_test_Bdf[,bags] <- c(rsq_test_vec, rep(NA,times=backpack_hyps[['maxit']]-length(rsq_test_vec)))
    
    
    
    # --- Storage for Predictions
    df_pred_Blist$Train$MACE[,paste0("B",bags)] <- c(vec_NA,MACE_pred_train)
    df_pred_Blist$Test$MACE[,paste0("B",bags)] <- MACE_pred_test
    
    
    # --- Storage for portfolio-weights
    MACE_weights_final <- data.frame('weight'= MACE_weights / sum(MACE_weights))
    rownames(MACE_weights_final) <- colnames(backpack_data[['Y_train']])
    
    
    for (aa in rownames(MACE_weights_final)){
      df_MACE_weights_Bdf[bags,which(colnames(df_MACE_weights_Bdf) == aa)] <- as.numeric(MACE_weights_final[aa,1])
    }
    
    
    # --- Storage for raw weights:
    MACE_weights_raw <- data.frame('weight'= MACE_betas[my_run,])
    
    rownames(MACE_weights_raw) <- c('c',colnames(backpack_data[['Y_train']]))
    
    
    for (aa in rownames(MACE_weights_raw)){
      df_MACE_weights_raw_Bdf[bags,which(colnames(df_MACE_weights_raw_Bdf) == aa)] <- as.numeric(MACE_weights_raw[aa,1])
    }
    
    # --- Next 'bags'
  
  
  
  
    # --- Pack the output
    backpack_out <- list('rsq_train_Bdf'=rsq_train_Bdf,
                         'rsq_test_Bdf'=rsq_test_Bdf,
                         'oob_MSE_rhs_vec'=oob_MSE_rhs_vec,
                         'hx_train_Bdf'=hx_train_Bdf,
                         'hx_test_Bdf'=hx_test_Bdf,
                         'hy_train_Bdf'=hy_train_Bdf,
                         'hy_test_Bdf'=hy_test_Bdf,
                         'fit_train_Bdf'=fit_train_Bdf,
                         'fit_test_Bdf'=fit_test_Bdf,
                         'my_run_Bvec'=my_run_Bvec,
                         'R2_train_Bvec'=R2_train_Bvec,
                         'R2_test_Bvec'=R2_test_Bvec,
                         'r1_hist_Blist'=r1_hist_Blist,
                         'r2_hist_Blist'=r2_hist_Blist,
                         'df_MACE_weights_Bdf'=df_MACE_weights_Bdf,
                         'df_MACE_weights_raw_Bdf'=df_MACE_weights_raw_Bdf,
                         'df_pred_Blist'=df_pred_Blist)
    
    
    list__B_out[[bags]] <- backpack_out
    
  }
  
  
  # --- Terminate the Cluster:
  #parallel::stopCluster(cl = my_cluster)
  
  
  
  ################################################################################################
  #
  #                    4)       Collect & Export
  #
  ################################################################################################
  
  
  
  # --- Collect the Results:
  for (b in 1:length(list__B_out)){
    
    rsq_train_Bdf[,paste0('B',b)] <- list__B_out[[b]][['rsq_train_Bdf']][,paste0('B',b)]
    rsq_test_Bdf[,paste0('B',b)]  <- list__B_out[[b]][['rsq_test_Bdf']][,paste0('B',b)]
    hx_train_Bdf[,paste0('B',b)]  <- list__B_out[[b]][['hx_train_Bdf']][,paste0('B',b)]
    hx_test_Bdf[,paste0('B',b)]   <- list__B_out[[b]][['hx_test_Bdf']][,paste0('B',b)]
    hy_train_Bdf[,paste0('B',b)]  <- list__B_out[[b]][['hy_train_Bdf']][,paste0('B',b)]
    hy_test_Bdf[,paste0('B',b)]   <- list__B_out[[b]][['hy_test_Bdf']][,paste0('B',b)]
    fit_train_Bdf[,paste0('B',b)] <- list__B_out[[b]][['fit_train_Bdf']][,paste0('B',b)]
    fit_test_Bdf[,paste0('B',b)]  <- list__B_out[[b]][['fit_test_Bdf']][,paste0('B',b)]
    
    my_run_Bvec[b]                <- list__B_out[[b]][['my_run_Bvec']][b]
    R2_train_Bvec[b]              <- list__B_out[[b]][['R2_train_Bvec']][b]
    R2_test_Bvec[b]               <- list__B_out[[b]][['R2_test_Bvec']][b]
    r1_hist_Blist[[b]]            <- list__B_out[[b]][['r1_hist_Blist']][[b]][my_run_Bvec[b]]
    r2_hist_Blist[[b]]            <- list__B_out[[b]][['r2_hist_Blist']][[b]][my_run_Bvec[b]]
    
    df_MACE_weights_Bdf[b,]       <- list__B_out[[b]][['df_MACE_weights_Bdf']][b,]
    df_MACE_weights_raw_Bdf[b,]   <- list__B_out[[b]][['df_MACE_weights_raw_Bdf']][b,]
    
    for (s in c('Train','Test')){
      for (m in names(df_pred_Blist[[s]])){
        df_pred_Blist[[s]][[m]][,paste0('B',b)] <- list__B_out[[b]][['df_pred_Blist']][[s]][[m]][,paste0('B',b)]
      }
    }
    
  }
  
  
  # --- Prepare for Export:
  backpack_out <- list('rsq_train_Bdf'=rsq_train_Bdf,
                       'rsq_test_Bdf'=rsq_test_Bdf,
                       'hx_train_Bdf'=hx_train_Bdf,
                       'hx_test_Bdf'=hx_test_Bdf,
                       'hy_train_Bdf'=hy_train_Bdf,
                       'hy_test_Bdf'=hy_test_Bdf,
                       'fit_train_Bdf'=fit_train_Bdf,
                       'fit_test_Bdf'=fit_test_Bdf,
                       'my_run_Bvec'=my_run_Bvec,
                       'R2_train_Bvec'=R2_train_Bvec,
                       'R2_test_Bvec'=R2_test_Bvec,
                       'r1_hist_Blist'=r1_hist_Blist,
                       'r2_hist_Blist'=r2_hist_Blist,
                       'df_MACE_weights_Bdf'=df_MACE_weights_Bdf,
                       'df_MACE_weights_raw_Bdf'=df_MACE_weights_raw_Bdf,
                       'df_pred_Blist'=df_pred_Blist)
  
  
  return(backpack_out)
  
  }
