
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
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%Part - 1
y_temp = eye(num_labels);
y = y_temp(y,:);

a1 = [ones(m,1) X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m,1) a2];
z3 = a2 * Theta2';
a3 =  sigmoid(z3);
h_theta = a3;
J = 0;
for i=1:m

    J = J + sum(-y(i,:).*log(h_theta(i,:)) - (1 - y(i,:)).*log(1 - h_theta(i,:)));
end
J = J/m;
sum = 0;
theta_1_original = Theta1;
theta_2_original = Theta2;
Theta1 = Theta1(:,2:end);
Theta2 = Theta2(:,2:end); 
%Regularizing
for a=1:size(Theta1,1)
  for b=1:size(X,2)
    sum = sum + Theta1(a,b)^2;
  end
end

for c=1:size(Theta2,1)
  for d=1:size(Theta2,2)
    sum = sum + Theta2(c,d)^2;
  end
end
J = J + (lambda/(2*m)) .* sum;

%Part - 2 Backpropagation
for t=1:m
  a_1 = X(t,:);
  a_1 = [1 a_1];
  z2 =  a_1 * theta_1_original';
  a_2 = sigmoid(z2);
  a_2 = [1 a_2];
  z3 = a_2 * theta_2_original';
  a_3 =  sigmoid(z3);
  %delta = zeros(size(a_3));
  %size(a_3)
  delta_3 = a_3 - y(t,:);
  %size(delta_3)
  delta_2 = (delta_3*theta_2_original).*sigmoidGradient([1 z2]);
  delta_2 = delta_2(2:end);
  Theta1_grad = Theta1_grad + delta_2'*a_1;
  Theta2_grad = Theta2_grad + delta_3'*a_2;
end
Theta1_grad = Theta1_grad./m;
Theta2_grad = Theta2_grad./m;

%Part - 3
Theta1_grad(:,2:input_layer_size+1) = Theta1_grad(:,2:input_layer_size+1) + lambda/m * theta_1_original(:,2:input_layer_size+1); 
Theta2_grad(:,2:hidden_layer_size+1) = Theta2_grad(:,2:hidden_layer_size+1) + lambda/m * theta_2_original(:,2:hidden_layer_size+1); 
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
% function [J grad] = nnCostFunction(nn_params, ...
                                   % input_layer_size, ...
                                   % hidden_layer_size, ...
                                   % num_labels, ...
                                   % X, y, lambda)
% %NNCOSTFUNCTION Implements the neural network cost function for a two layer
% %neural network which performs classification
% %   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
% %   X, y, lambda) computes the cost and gradient of the neural network. The
% %   parameters for the neural network are "unrolled" into the vector
% %   nn_params and need to be converted back into the weight matrices. 
% % 
% %   The returned parameter grad should be a "unrolled" vector of the
% %   partial derivatives of the neural network.
% %

% % Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% % for our 2 layer neural network
% Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 % hidden_layer_size, (input_layer_size + 1));

% Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 % num_labels, (hidden_layer_size + 1));


% % Setup some useful variables
% m = size(X, 1);

% X = [ones(m, 1) X];
% % You need to return the following variables correctly 
% J = 0;
% Theta1_grad = zeros(size(Theta1));
% Theta2_grad = zeros(size(Theta2));

% Theta1_Sq =  Theta1.^2;
% Theta2_Sq = Theta2.^2;

% Theta1_Sq(:,1) = Theta1_Sq(:,1).*0;
% Theta2_Sq(:,1) = Theta2_Sq(:,1).*0;

% X2 = hFunction(X,Theta1');
% n = size(X2,1);
% X2 = [ones(n,1), X2];
% X3 = hFunction(X2, Theta2');
% delta3 = zeros(size(X3));
% % ====================== YOUR CODE HERE ======================
% % Instructions: You should complete the code by working through the
% %               following parts.
% %
% % Part 1: Feedforward the neural network and return the cost in the
% %         variable J. After implementing Part 1, you can verify that your
% %         cost function computation is correct by verifying the cost
% %         computed in ex4.m
% %
% k = size(X3,2)

	% for K = 1:k
		% Y = (y == K);

		% J = J + sum((-Y)'*log(X3(:,K)) - (1 - Y)'*log(1- X3(:,K)));
		
		% delta3(:,K) = X3(:,K) - Y;
		% %p =-Y*log(X3[:,k]);%P = (Y'*log(X3[:;K]) - (1 - Y)'*log(1-X3[:;K]));
	% end

	% J = J + (lambda/2)*(sum(sum(Theta1_Sq)) + sum(sum(Theta2_Sq)));
	% J = J / m
% %
% % Part 2: Implement the backpropagation algorithm to compute the gradients
% %         Theta1_grad and Theta2_grad. You should return the partial derivatives of
% %         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
% %         Theta2_grad, respectively. After implementing Part 2, you can check
% %         that your implementation is correct by running checkNNGradients
% %
% %         Note: The vector y passed into the function is a vector of labels
% %               containing values from 1..K. You need to map this vector into a 
% %               binary vector of 1's and 0's to be used with the neural network
% %               cost function.
% %
% %         Hint: We recommend implementing backpropagation using a for-loop
% %               over the training examples if you are implementing it for the 
% %               first time.
% %
		
% %
% % Part 3: Implement regularization with the cost function and gradients.
% %
% %         Hint: You can implement this around the code for
% %               backpropagation. That is, you can compute the gradients for
% %               the regularization separately and then add them to Theta1_grad
% %               and Theta2_grad from Part 2.
% %
% for t=1:m
  % a_1 = X(t,:);
  % z2 =  a_1 * Theta1';
  % a_1 = [1 a_1];
  % a_2 = sigmoid(z2);
  % z3 = a_2 * Theta2';
  % a_3 =  sigmoid(z3);
  % a_2 = [1 a_2];
  % %delta = zeros(size(a_3));
  % %size(a_3)
  % delta_3 = a_3 - y(t,:);
  % %size(delta_3)
  % delta_2 = (delta_3*Theta2).*sigmoidGradient([1 z2]);
  % delta_2 = delta_2(2:end);
  % Theta1_grad = Theta1_grad + delta_2'*a_1;
  % Theta2_grad = Theta2_grad + delta_3'*a_2;
% end
% Theta1_grad = Theta1_grad./m;
% Theta2_grad = Theta2_grad./m;

% % for i = 1:m
	% % delta2 = (delta3(i,:) * Theta2)*sigmoidGradient(X2(i,:)');
	% % Theta1_grad = Theta1_grad + delta2'*X(i,:);
	% % Theta2_grad = Theta2_grad + delta3(i,:)'*X2(i,:);
% % end


% Theta1_grad = Theta1_grad / m;

% Theta2_grad = Theta2_grad/ m;



% % -------------------------------------------------------------

% % =========================================================================

% % Unroll gradients
% grad = [Theta1_grad(:) ; Theta2_grad(:)];


% end
