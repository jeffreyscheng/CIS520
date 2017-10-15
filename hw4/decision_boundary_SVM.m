function decision_boundary_SVM(X, y, model, grid_resolution, file_to_save)
% This function plots the decision boundary for your classifier with respect
% to the training data, labels given. Note that the arguement 

% X - m x 2 matrix of instances 
% y - m x 1 vector of labels (+1/-1)
% model - SVM model returned by libsvm
% resolution - 
%     resolution of the grid to be generated: an integer between 10 and
%     1000; for more accurate decision boundary, set this to a high value;
%     default value is 100
%file_to_save - the name of the file that you want to save the plot to

% Adapted from :
% http://www.mathworks.in/matlabcentral/fileexchange/34864-decision-boundary-using-svms/content/Decision%20Boundary%20using%20SVMs/visualizeBoundary.m
   
if(nargin < 4)
    grid_resolution = 100;
end


h = figure;


plot(X(y==1, 1), X(y==1, 2), '.b'), hold on;
plot(X(y==-1, 1), X(y==-1, 2), '.r'), hold on,

% Make classification predictions over a grid of values
x1plot = linspace(0, 7, grid_resolution)';
x2plot = linspace(0, 7, grid_resolution)';
[X1, X2] = meshgrid(x1plot, x2plot);
vals = zeros(size(X1));
for i = 1:size(X1, 2)
   this_X = [X1(:, i), X2(:, i)];
   vals(:, i) = svmpredict(ones(size(this_X,1),1), this_X, model, '-q');
end



% Plot the SVM boundary
hold on
contour(X1, X2, vals, [0 0], 'Color', 'k', 'LineWidth', 3);

xlim([1 7]);
ylim([1 7]);
xlabel('Feature 1');
ylabel('Feature 2');
legend('Positive Test Examples', 'Negative Test Examples', 'Classifier Boundary', 'Location','best');
hold off;

filename = strcat('Plots/',file_to_save);
title(strcat({'Decision Boundary for '}, file_to_save, ' kernel '),'Interpreter','latex','FontSize', 16, 'FontWeight', 'bold');
%axis([10^(-3) 1  0 0.7]);
set(gcf, 'PaperPositionMode', 'auto');
% save the file at the desired location
%saveas(gcf,filename,'fig');
print(h, '-dpng', filename);

