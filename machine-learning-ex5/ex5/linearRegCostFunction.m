function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

mat = (X*theta - y).^2;
J = sum(mat) / (2*m);
reg_mat = theta;
reg_mat(1,1) = 0;
reg_mat = reg_mat.^2;
J = J + ((lambda * sum(reg_mat))/(2 * m));

reg_mat = theta;
reg_mat(1,1) = 0;
reg_mat = (lambda * reg_mat) / m;

mat = (X*theta - y)' * X;
mat = mat' / m;
grad = mat + reg_mat;

% =========================================================================

grad = grad(:);

end
