function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%
% model= svmTrain(X, y, 0.01, @(x1, x2) gaussianKernel(x1, x2, 0.01)); 
% predictions = svmPredict(model, Xval);
% oldPredictionError = mean(double(predictions ~=yval));

% options = [0.01,0.03,0.1,0.3,1,3,10,30];

% for i = 1:8
	% C = options(i);
	% for k = 1:8
		% sigma = options(k);
		% model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
		
		% predictions = svmPredict(model, Xval);
		% PredictionError = mean(double(predictions ~=yval));
		% if(PredictionError < oldPredictionError)
			% best = [C,sigma,PredictionError];
			% oldPredictionError = PredictionError;
		% endif
	% end
% end
% You need to return the following variables correctly.
% C = 1;
% sigma = 0.3;
% cond = 1;
% C_ = 1;
% sigma_ = 0.3;
% C = 30;
% sigma = 3;

% model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma_)); 
% predictions = svmPredict(model, Xval);
% oldPredictionError = mean(double(predictions ~=yval));
% sigma_ = sigma_*3;
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
% best = [C,sigma_,oldPredictionError];

% while(C < 100)
	
	% while(sigma_<100)
		
		% model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma_)); 
		
		% predictions = svmPredict(model, Xval);
		% PredictionError = mean(double(predictions ~=yval));
		
		% if(PredictionError < oldPredictionError)
			% best = [C,sigma_,PredictionError];
			% oldPredictionError = PredictionError;
		% endif
		% sigma_ = sigma_ * 3;
	% endwhile
	% sigma_ = sigma;
	% C = 3 * C;azgXsl0vqLZ1Sd1o
% endwhile

% best
C = 1; % best(1)
sigma = 0.1;%best(2)

% =========================================================================

end
