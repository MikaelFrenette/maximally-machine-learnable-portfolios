######################################################################################################               		MAXIMALLY MACHINE LEARNABLE PORTFOLIOS
#
#			Goulet Coulombe, P. & Göbel, M. (2023)#####################################################################################################


% -------------------------------------------------------------------------------------------- %
			
				Folder Structure

% -------------------------------------------------------------------------------------------- %

	- MACE_MMLP_daily.R:	this is the main file

			

	- daily__MACE_paper.csv:	this is the data file with around 200 stocks. In late 2022, a snapshot of the NASDAQ was taken and for each stock the market cap at the end of 2016 was calculated. The file contains the ~200 most-capitalized stocks, sorted in decreasing order. (--> yes, not totally free of hindsight bias)

	- 001_functions/MACE_daily.R:	this is the actual MACE algorithm (Algorithm 1 in the paper), which computes the maximally machine-learnable portfolio


 	- 001_functions/Trading_MV_daily.R:	this is a function for trading the MACE portfolio, i.e. deploying the volatility-timing strategy

			
	- 001_functions/auxiliaries.R:	several auxiliary functions, with some not being in use anymore…


% -------------------------------------------------------------------------------------------- %

			Remarks on the main file 	MACE_MMLP_daily.R

% -------------------------------------------------------------------------------------------- %
			

The file contains four sections:

                 1.    MMLP - Estimation                 2.    Trading
		 3.    Post-Trading Evaluation
		 4.    Plotting of Cumulative Returns

Each section may contain a subsection called 'USER INTERACTION'. Here you can set your own hyperparameters,



In section 'Trading', you can also bring in your own portfolios. You only need to insert:

	- line 348:	portfolio weights for each stock in 'Y'
	- line 352:	time-series of out-of-sample predictions for your portfolio returns


					