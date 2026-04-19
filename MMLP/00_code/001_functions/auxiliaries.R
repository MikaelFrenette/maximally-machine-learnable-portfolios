

# ============================ Update Tickers ============================ #

get__ticUPDATE <- function(map,MACE_tics){
  
  tics_replace <- MACE_tics[which(MACE_tics %in% rownames(map))]
  MACE_tics[which(MACE_tics %in% tics_replace)] <- map[tics_replace,'tic_new']
  print(paste0('Updated Tickers: ',tics_replace))
  
  return(MACE_tics)
}

# ================================================================================================================ #

block.sampler <- function(X,sampling_rate,block_size,num.tree){
  
  inbag=list()
  inbag2=list()
  
  for(j in 1:num.tree){
    sample_index <- c(1:nrow(X))
    groups<-sort(base::sample(x=c(1:(length(sample_index)/block_size)),size=length(sample_index),replace=TRUE))
    rando.vec <- rexp(rate=1,n=length(sample_index)/block_size)[groups] +0.1
    chosen.ones.plus<-rando.vec
    rando.vec<-which(chosen.ones.plus>quantile(chosen.ones.plus,1-sampling_rate))
    chosen.ones.plus<-sample_index[rando.vec]
    
    boot <- c(sort(chosen.ones.plus))                         
    oob <- c(1:nrow(X))[-boot]   
    count = 1:nrow(X)
    inbagj = I(is.element(count,boot))
    inbag[[j]] = as.numeric(inbagj)
  }
  
  for(j in 1:num.tree){
    sample_index <- c(1:nrow(X))
    groups<-sort(base::sample(x=c(1:(length(sample_index)/block_size)),size=length(sample_index),replace=TRUE))
    rando.vec <- rexp(rate=1,n=length(sample_index)/block_size)[groups] +0.1
    chosen.ones.plus<-rando.vec
    rando.vec<-which(chosen.ones.plus>quantile(chosen.ones.plus,1-sampling_rate))
    chosen.ones.plus<-sample_index[rando.vec]
    
    boot <- c(sort(chosen.ones.plus))                         
    oob <- c(1:nrow(X))[-boot]   
    count = 1:nrow(X)
    inbagj = I(is.element(count,boot))
    inbag2[[j]] = as.numeric(inbagj)
  }
  
  return(list(inbag1=inbag,inbag2=inbag2))
}

# ============================ Prepare MARX Features ============================ #
get__MARX <- function(series_oos,series_ins=NULL,N_lags){
  
  if (length(N_lags) == 1){
    N_lags <- 1:N_lags
  }
  
  # --- --- --- Create the Moving-Average Rotations of 'X':
  mace_marx <- matrix(sapply(N_lags, function(x) lag(series_oos,x)),
                      nrow=length(series_oos),ncol=length(N_lags),dimnames = list(c(),paste0('MARX',N_lags)))
  
  # --- --- --- Fill the missing with in-sample values:
  if (!is.null(series_ins)){
    for (ii in N_lags){
      mace_marx[1:ii,which(N_lags == ii)] <- tail(series_ins, ii)
    }
  }
  
  rownames(mace_marx) <- names(series_oos)
  
  mace_marx[is.na(mace_marx)] <- 0
  
  mace_marx <- t(apply(mace_marx,1,cumsum))
  
  return(mace_marx)
  #return(mace_marx[complete.cases(mace_marx),])
}


# ============================ Rolling Mean (Forecast) ============================ #
get__rollMean <- function(series,lookback,h=1){
  
  yhat <- rep(NA,times=length(series))
  for (t in (lookback+h):length(series)){
    yhat[t] <- mean(series[(t-lookback):(t-h)])
  }
  return(yhat)
}



# ============================ Downloading Latest Market-Data ============================ #
getData__Yahoo <- function(MACE_tics,startDate='2000-02-03'){
  
  library(quantmod)
  library(lubridate)
  
  
  # -------------------------------------- Get Prices -------------------------------------- #
  
  prices <- getSymbols(MACE_tics[1], src='yahoo', from='2000-01-01', periodicity='daily',auto.assign = F)
  prices <- data.frame('date'=as.Date(index(prices)),'val'=prices[,paste0(gsub('\\^','',MACE_tics[1]),'.Adjusted')])
  colnames(prices)[2] <- gsub('\\^','',MACE_tics[1])
  
  # --- For Compatibility:
  rownames(prices) <- c(1:nrow(prices))
  
  
  pb <- txtProgressBar(min=0,max=length(MACE_tics)-1,initial=1, style=3)
  i <- 1
  for (tic in MACE_tics[-1]){
    
    setTxtProgressBar(pb,i)
    
    # --- Get the Price data:
    prices_tic <- getSymbols(tic, src='yahoo', from='2000-01-01', periodicity='daily',auto.assign = F)
    prices_tic <- data.frame('date'=as.Date(index(prices_tic)),'val'=prices_tic[,paste0(gsub('\\^','',tic),'.Adjusted')])
    colnames(prices_tic)[2] <- gsub('\\^','',tic)
    
    # --- Merge with 'prices'
    prices <- merge(prices, prices_tic, by='date', all.x=TRUE)
    
    i <- i+1
    
  }
  close(pb)
  
  
  
  # -------------------------------------- Transform to Returns -------------------------------------- #
  returns <- data.frame(cbind(date=prices['date'],matrix(NA,nrow=nrow(prices),ncol=ncol(prices)-1, dimnames = list(c(),colnames(prices)[-1]))))
  returns[,-1] <- apply(prices[,-1],2, function(x) c(NA,log(x[2:nrow(prices)])-log(x[1:(nrow(prices)-1)])))
  #returns[,-1] <- apply(prices[,-1],2, function(x) c(NA,x[2:nrow(prices)] / x[1:(nrow(prices)-1)] - 1))
  
  
  
  # ------------------------------- Export ------------------------------- #
  returns <- returns[returns['date'] >= startDate,]
  
  return(returns)
  #return(list('returns'=returns,'prices'=prices))
  
  
}



# ================================================================================================================ #


roll_mean <- function(series_oos,series_ins,lookback,h){
  roll_mean_store <- c()
  if (length(series_oos) == 0){
    for (ll in 1){
      if (h > ll){
        roll_mean_store <- c(roll_mean_store,mean(tail(c(series_ins[0:(length(series_ins)-h)]),lookback)))
      } else {
        roll_mean_store <- c(roll_mean_store,mean(tail(c(series_ins,series_oos[0:(ll-h)]),lookback)))
      }
    }
  } else {
    for (ll in 1:length(series_oos)){
      roll_mean_store <- c(roll_mean_store,mean(tail(c(series_ins,series_oos)[1:(length(series_ins)+ll-h)],lookback)))
    }
  }
  
  return(roll_mean_store)
}


# ============================ Position-Sizing via Mean-Variance Optimization ============================ #

MeanVariance <- function(return_oos_true,return_oos_pred,return_ins,params){
  
  # --- Unpack your parameters
  lookback <- params[["lookback"]]
  gamma_mv <- params[["gamma"]]
  alpha_mv <- params[["alpha"]]
  port_pos_max <- params[["MV_pos_max"]]
  port_pos_min <- params[["MV_pos_min"]]
  vol_model <- params[["vol_model"]]
  H <- params[['H']]
  
  position_oos <- rep(0,times=length(return_oos_pred))
  
  for (tt in 1:length(return_oos_pred)){
    # --- Prediction for period 'tt'
    return_pred_tt <- return_oos_pred[tt]
    # --- Prepare a Volatility-Forecast:
    if (vol_model == 'ewma'){
      # --- Calculate the Exponentially-Weighted-Moving-Average of the variance of the portfolio
      sigma2_tt <- tail(ewma(tail(c(return_ins,return_oos_true)[1:(length(return_ins)+tt-H)],lookback), alpha_mv),1)
    } else if (vol_model == 'albaMA'){
      sigma2_tt <- AlbaMA(c(return_ins,return_oos_true)[1:(length(return_ins)+tt-H)]^2)
    }
    # --- Calculate the optimal weight
    weight_tt <- (1/gamma_mv)*(return_pred_tt/sigma2_tt)
    # --- Keep the weight within reasonable bounds
    position_oos[tt] <- max(c(min(weight_tt,port_pos_max),port_pos_min))
  }
  
  # --- Calculate returns & store positions
  out <- list('position'=position_oos, 'return'=position_oos * return_oos_true)
  
  return(out)
  
  
}



AlbaMA <- function(series, h=1){
  
  # --- Prepare the Feature-Matrix:
  mat <- data.frame(cbind(y=series,X=c(1:length(series))))
  
  # --- Fit the RandomForest:
  r__albaMA <- ranger(y ~ ., data=mat)
  
  # --- Make the Forecast:
  yhat_oos <- predict(r__albaMA, data = data.frame('X'=c((length(series)+1):(length(series)+h))))$predictions
  
  return(yhat_oos)
  
}



# ================= New Features + Own Vol-Estimator (Exponential & Non-Exponential possible) ================= #


vol_expWEIGHT <- function(series,lmbda,window,I_want_OOS=FALSE){
  
  # --- BEWARE: the value at 't' will be based on data up until 't-1'
  # --- BEWARE: if 'window == NULL', at each step the historical sample variance will be calculated
  
  # --- Initialize the vector
  vol <- rep(NA,length(series)+I_want_OOS)
  
  # --- Was the sample-variance requested?
  if (is.null(window)){
    
    # --- CONVENTION: We use 251-Trading-Days for initialization
    
    # --- At each time-step, the sample variance is used:
    for (t in (252):(length(series)+I_want_OOS)){
      vol[t] <- var(series[1:(t-1)])
    }
    
  } else {
    
    # --- Run over all time-steps:
    for (t in (window+1):(length(series)+I_want_OOS)){
      vol[t] <- var(series[(t-window):(t-1)])
      
      # --- In case 'lmbda != 1', there will be some exponential weighting involved
      for (w in 1:window){
        vol[t] <- lmbda * vol[t] + (1-lmbda) * (series[t-window-1+w]^2)
      }
      # --- Next: 't'
    }
  }
  
  # --- Export the volatility!
  return(vol^0.5)
  
}

# ================================================================================================================ #


build__addFeatures <- function(backpack_data,I_want_OOS=FALSE){
  
  
  # --- I_want_OOS:   this is for live-trading. You take into account the very last observation, 
  #                   to create features for TOMMORROW's prediction!
  
  # --- Want to build additional features?
  # --- NULL: no additional features
  # --- --- 'lookback' = c(NULL) ---> historical volatility is used (burn-in: 251 observations)
  # --- --- 'estimator' = c('exp__1) ---> no exponential weighting is applied
  # ATTENTION: all estimates already come as Lag-1 ! Hence, the estimate at 't' is based on information up until time 't-1'
  
  
  # --- Off-load some stuff
  specs_file <- backpack_data[['specs_file']]
  specs_idx <- backpack_data[['specs_idx']]
  pred_y_test <- backpack_data[['Ytest_pred']]
  pred_y_train <- backpack_data[['Ytrain_pred']]
  
  
  # --- Collect the Feature-Specification
  backpack_data[['build_features']] <- list()
  
  # --- Build the backpack for each requested feature-constructor:
  add_features <- strsplit(specs_file[specs_idx,'build_features'],',')[[1]]
  for (f in add_features){
    if (f == 'vol'){
      backpack_data[['build_features']][[f]] <- list('lags'=eval(parse(text=specs_file[specs_idx,'bf__lags'])),
                                                     'lookback'=eval(parse(text=specs_file[specs_idx,'bf__lookback'])),
                                                     'estimator'=strsplit(specs_file[specs_idx,'bf__estimator'],',')[[1]]
      )
    }
  }
  
  
  
  # --- Instantiate the matrix for the new features
  newFEAT <- matrix(nrow=length(pred_y_test)+I_want_OOS,ncol=0)
  
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
        out_vol <- vol_expWEIGHT(series_test,lmbda=e_lmbda,window=w,I_want_OOS)
        
        # --- REMEMBER: the estimates in 'out_vol' already come as 1-period lagged, i.e. are forecasts for day 't' based on information up until 't-1'
        newFEAT <- cbind(newFEAT,matrix(out_vol[(w+1):(length(out_vol)-l+1)],
                                        dimnames=list(c(),paste0('L',l,'_vol__LB',w,'_exp',e_lmbda))))
        # --- Next: 'l'
      }
      # --- Next: 'w'
    }
    # --- Next: 'e'
  }
  
  
  # --- Export the Matrix of additional Features:
  return(newFEAT)
  
}


# ============================ Exponentially-Weighted Moving-Average ============================ #


ewma <- function(x, lambda) {
  
  # --- If x is a vector
  if (is.null(dim(x))){
    # x = vector of returns
    if (length(x) <= 1 || !all(!is.na(x)) || !is.numeric(x)) {
      stop("A numeric vector of length > 1 and without NAs must be passed to",
           " 'x'.")
    }
    if (length(lambda) != 1 || is.na(lambda) || !is.numeric(lambda) ||
        lambda < 0 || lambda >= 1) {
      stop("The argument 'lambda' must be a single non-NA double value with ",
           "0 < lambda < 1.")
    }
    n <- length(x)
    vol <- rep(NA, n)
    vol[1] <- stats::var(x)
    for (i in 2:n) {
      vol[i] <- lambda * vol[i - 1] + (1 - lambda) * x[i - 1]^2
    }
    
  } else {
    
    n <- nrow(x)
    vol <- rep(list(c()), n)
    vol[[1]] <- stats::cov(x)
    for (i in 2:n) {
      ret_i <- x[(i - 1),]
      vol[[i]] <- lambda * vol[[i - 1]] + (1 - lambda) * (ret_i %*% t(ret_i))
    }
    
  }  
  
  return(vol)
}


# ============================================== Some Metrics ============================================== #

returnA <- function(series,freq=250){
  # --- freq: in terms of annual observations (i.e. daily=250; monthly = 12; quarterly = 4; yearly=1)
  return(mean(series)*freq)
}

volA <- function(series,freq=250){
  # --- freq: in terms of annual observations (i.e. daily=250; monthly = 12; quarterly = 4; yearly=1)
  return(sd(series)* freq^0.5)
}

sharpe_ratio <- function(series,freq=250, rf=0, TC=0){
  mu <- returnA(series,freq) - TC
  vol <- volA(series,freq)
  return((mu-rf)/vol)
}


maxDD <- function(series){
  pnl <- exp(cumsum(series))
  mdd <- min(pnl/cummax(pnl) - 1)
  
  return(mdd)
}



calmar_ratio <- function(series,freq=250){
  # --- freq: in terms of annual observations (i.e. daily=250; monthly = 12; quarterly = 4; yearly=1)
  
  # --- Get the Maximum Drawdown:
  mdd <- -maxDD(series)
  
  if (abs(mdd) < 1e-10){
    return(NA)
  } else {
    return(returnA(series,freq)/mdd)
  }
  
}

sortino_ratio <- function(series, freq = 250, target = 0) {
  # --- freq: in terms of annual observations (i.e. daily=250; monthly = 12; quarterly = 4; yearly=1)
  downside <- pmin(series - target, 0)
  downside_dev <- sqrt(mean(downside^2)) * sqrt(freq)
  
  if (downside_dev < 1e-10) {
    return(NA)
  } else {
    return(returnA(series, freq) / downside_dev)
  }
}



R2 <- function(y,yhat,ybase){
  SSR <- sum((y-yhat)^2)
  SST <- sum((y-ybase)^2)
  return(1 - SSR/SST)
}



# ==================================== Turnover: Gu, Kelly, Xiu (2020) ========================================== #


func_turnoverGKX <- function(series,weights){
  
  if (class(series)[1] %in% c('matrix','data.frame')){
    TT <- nrow(weights)
    turnover <- c()
    
    for (tt in 2:TT){
      
      turnover <- c(turnover,sum(abs(as.numeric(weights[tt,]) - (as.numeric(weights[tt-1,])*as.numeric(1+series[tt,]) / (1 + sum(as.numeric(weights[tt-1,])*as.numeric(series[tt,]), na.rm=T)) )),na.rm=T))
    }
  } else {
    TT <- length(weights)
    turnover <- c()
    
    for (tt in 2:TT){
      
      turnover <- c(turnover,abs(weights[tt] - (weights[tt-1]*(1+series[tt]) / (1 + sum(weights[tt-1]*series[tt], na.rm=T)) )))
    }
  }
  
  out <- list('turnover'=sum(turnover, na.rm=T) / TT, 'turnover_ts' = turnover)
  return(out)
}


# ==================================== Covariance Shrinkage Estimators: Ledoit & Wolf ========================================== #


rep.row <- function(x, n){
  matrix(rep(x, each = n), nrow = n)
}

rep.col <- function(x, n){
  matrix(rep(x, times = n), ncol = n, byrow = F)
}


do__CovShrinkage_LW04 <- function(Y, k = 0) {
  dim.Y <- dim(Y)
  N <- dim.Y[1]
  p <- dim.Y[2]
  if (k < 0) {    # demean the data and set k = 1
    Y <- scale(Y, scale = F)
    k <- 1
  }
  n <- N - k    # effective sample size
  c <- p / n    # concentration ratio
  sample <- (t(Y) %*% Y) / n   
  
  # compute shrinkage target
  samplevar <- diag(sample)
  sqrtvar <- sqrt(samplevar)
  rBar <- (sum(sample / outer(sqrtvar, sqrtvar)) - p) / (p * (p - 1))
  target <- rBar * outer(sqrtvar, sqrtvar)
  diag(target) <- samplevar
  
  
  # estimate the parameter that we call pi in Ledoit and Wolf (2003, JEF)
  Y2 <- Y^2
  sample2 <- (t(Y2) %*% Y2) / n   
  piMat <- sample2 - sample^2
  pihat <- sum(piMat)
  
  # estimate the parameter that we call gamma in Ledoit and Wolf (2003, JEF)
  gammahat <- norm(c(sample - target), type = "2")^2
  
  # diagonal part of the parameter that we call rho 
  rho_diag <- sum(diag(piMat))
  
  # off-diagonal part of the parameter that we call rho 
  term1 <- (t(Y^3) %*% Y) / n;
  term2 <- rep.row(samplevar, p) * sample;
  term2 <- t(term2)
  thetaMat <- term1 - term2
  diag(thetaMat) <- 0
  rho_off <- rBar * sum(outer(1/sqrtvar, sqrtvar) * thetaMat)
  
  # compute shrinkage intensity
  rhohat <- rho_diag + rho_off
  kappahat <- (pihat - rhohat) / gammahat
  shrinkage <- max(0, min(1, kappahat / n))
  
  # compute shrinkage estimator
  sigmahat <- shrinkage * target + (1 - shrinkage) * sample
  
  
  
  
  return(list('shrinkage_COV'=sigmahat,'shrinkage'=shrinkage,'target_COV'=target))
}




do__CovShrinkage_LW03 <- function(Y, k = 0) {
  dim.Y <- dim(Y)
  N <- dim.Y[1]
  p <- dim.Y[2]
  if (k < 0) {    # demean the data and set k = 1
    Y <- scale(Y, scale = F)
    k <- 1
  }
  n <- N - k    # effective sample size
  c <- p / n    # concentration ratio
  sample <- (t(Y) %*% Y) / n   
  
  # compute shrinkage target
  Ymkt <- matrix(apply(Y, 1, mean),ncol = 1)
  covmkt <- as.vector((t(Ymkt) %*% Y) / n)
  varmkt <- c((t(Ymkt) %*% Ymkt) / n)
  target <- outer(covmkt, covmkt) / varmkt
  diag(target) <- diag(sample)
  
  # estimate the parameter that we call pi in Ledoit and Wolf (2003, JEF)
  Y2 <- Y^2
  sample2 <- (t(Y2) %*% Y2) / n   
  piMat <- sample2 - sample^2
  pihat <- sum(piMat)
  
  # estimate the parameter that we call gamma in Ledoit and Wolf (2003, JEF)
  gammahat <- norm(c(sample - target), type = "2")^2
  
  # diagonal part of the parameter that we call rho 
  rho_diag <- sum(diag(piMat))
  
  # off-diagonal part of the parameter that we call rho 
  temp <- Y * rep.col(Ymkt, p)
  v1 <- (1/n) * t(Y2) %*% temp - rep.col(covmkt, p) * sample
  roff1 <- sum(v1 * t(rep.col(covmkt, p))) / varmkt - sum(diag(v1) * covmkt) / varmkt
  v3 <- (1/n) * t(temp) %*% temp - varmkt * sample
  roff3 <- sum(v3 * (covmkt %*% t(covmkt))) / varmkt^2 - sum(diag(v3) * covmkt^2) / varmkt^2
  rho_off <- 2 * roff1 - roff3
  
  # compute shrinkage intensity
  rhohat <- rho_diag + rho_off
  kappahat <- (pihat - rhohat) / gammahat
  shrinkage <- max(0, min(1, kappahat / n))
  
  # compute shrinkage estimator
  sigmahat <- shrinkage * target + (1 - shrinkage) * sample
  
  
  return(list('shrinkage_COV'=sigmahat,'shrinkage'=shrinkage,'target_COV'=target))
}


# ==================================== For Plotting ========================================== #

theme_Publication <- function(base_size=30, base_family="Arial") {
  (ggthemes::theme_foundation(base_size=base_size, base_family=base_family)
   + theme(
     plot.title = element_text(face = "bold",
                               size = rel(1.3), hjust = 0.5),
     text = element_text(),
     plot.background = element_rect(colour = NA),
     panel.border = element_rect(colour = 'black'),
     axis.title = element_text(face = "bold", size = rel(1)),
     axis.title.y = element_text(angle=90, vjust = 2),
     axis.title.x = element_text(angle=0, vjust = -0.2),
     axis.text = element_text(),
     axis.line = element_line(colour="black"),
     axis.ticks = element_line(),
     panel.spacing    = unit(0.5, "lines"),
     panel.background = element_rect(fill = "white", colour = "black"),
     panel.grid.major = element_line(colour="#f0f0f0"),
     panel.grid.minor = element_line(colour="#f0f0f0"),
     legend.key = element_rect(colour = NA),
     legend.position = "bottom",
     legend.direction = "horizontal",
     legend.key.size= unit(0.6, "cm"),
     legend.margin = margin(-20,5,5,-20),
     legend.box.margin = margin(-5,-5,-5,-5),
     legend.title = element_text(face="italic"),
     plot.margin = unit(c(5,5,5,5),"mm"),
     strip.background = element_rect(colour="#f0f0f0",fill="#f0f0f0"),
     strip.text = element_text(face="bold")
   ))
}

break_function = function(x) {
  
  library(DescTools)
  
  # if(x[2]*x[1]<0){ #gotta include zero
  
  #target=7
  #for(i in 3:20){
  i=6
  init = seq(from = x[1], to = x[2], length.out = i)
  ll = abs(init[2]-init[1])
  candi = c(0.005,0.01,0.05,0.1,0.2,0.25,0.5,1,2,2.5,5,10)
  gapchoice=candi[which.min(abs(ll-candi))]
  xnew = RoundTo(x,gapchoice)
  if(xnew[1]>x[1]){xnew[1]-gapchoice}
  if(xnew[2]<x[2]){xnew[2]+gapchoice}
  thisshit=seq(from = xnew[1], to = xnew[2], by = gapchoice)
  #if(length(thisshit)==target){finalshit=thisshit}
  #print(length(thisshit))
  #}
  # }
  #unique(RoundTo(sort(append(0,)),multiple=0.05))
  return(thisshit)
}

get_Recessions <- function(region,freq=12, start_year=1960, end_year){
  
  # --- region: US, EA, CAN, QC
  # --- freq: 1=annual; 4=quarterly; 12=monthly
  
  if (region=='US'){
    NBER_Recessions <- nber(start_year,end_year,freq=freq,rec_ind='USREC')
  } else if (my_region == 'EA'){
    NBER_Recessions <- data.frame("start" = c(2008.25,2011.75,2020),
                                  "end"   = c(2009.25,2013,2020.25))
  } else if (my_region %in% c('CAN','QC')){
    NBER_Recessions <- data.frame("start" = c(2008.75,2020),
                                  "end"   = c(2009.25,2020.25))
  }
  
  
  NBER_Recessions_dates <- data.frame(matrix(NA,nrow=nrow(NBER_Recessions),ncol=ncol(NBER_Recessions),
                                             dimnames=list(c(),colnames(NBER_Recessions))))
  
  for (rr in 1:nrow(NBER_Recessions)){
    for (cc in 1:ncol(NBER_Recessions)){
      NBER_Recessions_dates[rr,cc] <- format(date_decimal(NBER_Recessions[rr,cc]), "%Y-%m-%d")
      
    }
  }
  
  NBER_Recessions_dates$start <- floor_date(as.Date(NBER_Recessions_dates$start, format = "%Y-%m-%d"), "month")
  NBER_Recessions_dates$end <- floor_date(as.Date(NBER_Recessions_dates$end, format = "%Y-%m-%d"), "month")
  
  for (rrr in 1:nrow(NBER_Recessions_dates)){
    for (cc in 1:ncol(NBER_Recessions_dates)){
      NBER_Recessions_dates[rrr,cc] <- as.Date(NBER_Recessions_dates[rrr,cc]) - months(1)
    }
  }
  
  return(NBER_Recessions_dates)
  
}

nber <- function(start_year,end_year,freq,rec_ind='USREC'){
  
  library(quantmod)
  library(lubridate)
  
  #NBER_Rec <- read.csv(file.path("/Users/maximilian/Desktop/Networks/Part 3_ConfidenceRealFinancialEconomy/Data", "NBER_Recession.csv"),header = TRUE)
  
  # --- Download Recession-dates from FRED
  raw <- data.frame(getSymbols(rec_ind, src = 'FRED', auto.assign = F))
  NBER_Rec <- data.frame("DATE"=rownames(raw),"USREC"=raw[rec_ind])
  
  NBER_df <- data.frame(matrix(NA, nrow = (end_year-start_year + 1)*12, ncol = 3))
  
  years <- seq(start_year,end_year,1)
  years_repeated <- 0
  for (repeate in 1:length(years)){
    years_repeated[(length(years_repeated)+1):(length(years_repeated)+12)] <- rep(years[repeate], times = 12)
  }
  years_repeated <- years_repeated[-1]
  
  NBER_df[,1] <- years_repeated
  NBER_df[,2] <- rep(seq(1,12,1), times=length(years))
  #----assigning S&P and ConSent
  NBER_df[,3] <- NBER_Rec[which(NBER_Rec[,1]==paste0(start_year,"-01-01")):which(NBER_Rec[,1]==paste0(end_year,"-12-01")), 2]
  NBER_df[,4] = seq(start_year + (1-1)/12, end_year + (12-1)/12, by = 1/12)
  colnames(NBER_df) <- c("Year", "Month", "NBER_Rec", "Date")
  
  #create start-end data frame for NBER
  NBER_dates <- data.frame(matrix(NA,1,2))
  colnames(NBER_dates) <- c("start", "end")
  
  # --- Define START- and END-Dates
  rec_months <- which(NBER_df$NBER_Rec == 1)
  
  if (length(rec_months) > 0){
    NBER_dates[dim(NBER_dates)[1]+1,1] <- NBER_df$Date[rec_months[1]]
    for (rr in 2:length(rec_months)){
      
      if (rec_months[rr] == rec_months[rr-1] + 1){
        
        # --- At the end of the loop, set the end date as follows:
        if (rr == length(rec_months)){
          NBER_dates[dim(NBER_dates)[1],2] <- NBER_df$Date[rec_months[rr]]
        }
        
      } else {
        # --- Set the end of the PREVIOUS Recession
        NBER_dates[dim(NBER_dates)[1],2] <- NBER_df$Date[rec_months[rr-1]]
        # --- Set the start of the NEW Recession
        NBER_dates[dim(NBER_dates)[1]+1,1] <- NBER_df$Date[rec_months[rr]]
        
        # --- At the end of the loop, set the end date as follows:
        if (rr == length(rec_months)){
          NBER_dates[dim(NBER_dates)[1],2] <- NBER_df$Date[rec_months[rr]]
        }
        
      }
      
    }
    
    # --- The old and not entirely bug-free procedure of defining START- and END-Dates
    #     (The algorithm couldn't work with some end-dates and would break)
    if (1==2){
      counter <- 1
      for (nber in counter:(dim(NBER_df)[1]-1)){
        nber <- counter
        if ((NBER_df$NBER_Rec[nber] == 1)==TRUE){
          NBER_dates[dim(NBER_dates)[1]+1,1] <- NBER_df$Date[nber]
          new_row <- min(which(NBER_df$NBER_Rec[(nber+1):dim(NBER_df)[1]] < 1)) + nber
          NBER_dates[dim(NBER_dates)[1],2] <- NBER_df$Date[(new_row-1)]
          counter <- new_row
        } else if (counter < dim(NBER_df)[1]) {
          counter <- counter + 1
        }
      }
    }
    NBER_dates <- NBER_dates[-1,]
    NBER_dates <- data.frame(start = c(NBER_dates$start), end = c(NBER_dates$end))
    
    if (freq %in% c(1,4)){
      #   Transform into Quarters
      NBER_dates_Q <- NBER_dates
      for (r in 1:nrow(NBER_dates)){
        for (c in 1:ncol(NBER_dates)){
          if (NBER_dates[r,c] - as.numeric(substr(NBER_dates[r,c],start=1,stop=4)) < 0.25){
            NBER_dates_Q[r,c] <- as.numeric(paste0(substr(NBER_dates[r,c],start=1,stop=4),".00"))
          } else if (NBER_dates[r,c] - as.numeric(substr(NBER_dates[r,c],start=1,stop=4)) >= 0.25 && (NBER_dates[r,c] - as.numeric(substr(NBER_dates[r,c],start=1,stop=4)) < 0.5)){
            NBER_dates_Q[r,c] <- as.numeric(paste0(substr(NBER_dates[r,c],start=1,stop=4),".25"))
          } else if (NBER_dates[r,c] - as.numeric(substr(NBER_dates[r,c],start=1,stop=4)) >= 0.5 && (NBER_dates[r,c] - as.numeric(substr(NBER_dates[r,c],start=1,stop=4)) < 0.75)){
            NBER_dates_Q[r,c] <- as.numeric(paste0(substr(NBER_dates[r,c],start=1,stop=4),".50"))
          } else if (NBER_dates[r,c] - as.numeric(substr(NBER_dates[r,c],start=1,stop=4)) >= 0.75){
            NBER_dates_Q[r,c] <- as.numeric(paste0(substr(NBER_dates[r,c],start=1,stop=4),".75"))
          }
        }
      }
      NBER_dates <- NBER_dates_Q
    }
  } else {
    NBER_dates <- data.frame()
  }
  return(NBER_dates)
  
}
