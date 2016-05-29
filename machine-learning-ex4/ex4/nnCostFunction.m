function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


% Setup some useful variables
m = size(X, 1);

X = [ones(m, 1) X];
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

Theta1_Sq =  Theta1.^2;
Theta2_Sq = Theta2.^2;

Theta1_Sq(:,1) = Theta1_Sq(:,1).*0;
Theta2_Sq(:,1) = Theta2_Sq(:,1).*0;

X2 = hFunction(X,Theta1');
n = size(X2,1);
X2 = [ones(n,1), X2];
X3 = hFunction(X2, Theta2');
delta3 = zeros(size(X3));
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
k = size(X3,2)

	for K = 1:k
		Y = (y == K);

		J = J + sum((-Y)'*log(X3(:,K)) - (1 - Y)'*log(1- X3(:,K)));
		
		delta3(:,K) = X3(:,K) - Y;
		%p =-Y*log(X3[:,k]);%P = (Y'*log(X3[:;K]) - (1 - Y)'*log(1-X3[:;K]));
	end

	J = J + (lambda/2)*(sum(sum(Theta1_Sq)) + sum(sum(Theta2_Sq)));
	J = J / m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
		
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
delta1  = zeros(size(Theta1));
delta2 = zeros(size(Theta2));

delta2_final = (((delta3*Theta2)' * X3)') / m;


for i=1:m
	delta2_temp = ((delta3(i,:) * Theta2)' * X3(i,:))';
	delta2_temp = delta1_temp(2:end,:);
	delta1_temp = zeros(size(Theta1));
	for K=1:k
		
	end
	delta2 = delta2 + delta2_temp;
	delta1 = delta1 + delta1_temp;
	
end

%delta2 = (delta2(2:end,:));

Theta1_grad = delta2_final;

Theta2_grad = delta2 / m;



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
