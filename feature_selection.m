% Load the dataset
data = readtable('UCI HAR Dataset/train/X_train.txt');
labels = readtable('UCI HAR Dataset/train/y_train.txt');

% Convert to matrix
data = table2array(data);
labels = table2array(labels);


%% Perform Feature Selection Using Mutual Information

% Calculate mutual information for each feature
mi = zeros(1, size(data, 2));
for i = 1:size(data, 2)
    mi(i) = mutual_information(data(:, i), labels);
end

% Select the top k features based on mutual information
k = 20; % Select top 20 features
[~, sorted_idx] = sort(mi, 'descend');
selected_features = data(:, sorted_idx(1:k));

% Display the top selected features
disp('Selected feature indices:');
disp(sorted_idx(1:k));

%% Train a Model Using the Selected Features
% Split data into training and testing sets (80% train, 20% test)
cv = cvpartition(labels, 'Holdout', 0.2);
X_train = selected_features(training(cv), :);
y_train = labels(training(cv));
X_test = selected_features(test(cv), :);
y_test = labels(test(cv));

% Train a multi-class SVM model using fitcecoc
SVMModel = fitcecoc(X_train, y_train);


% Test the model
y_pred = predict(SVMModel, X_test);


%% Evaluate the Model Performance

% Calculate accuracy
accuracy = sum(y_pred == y_test) / numel(y_test);

% Confusion matrix
conf_mat = confusionmat(y_test, y_pred);

% Calculate precision, recall, F1-score
tp = conf_mat(1, 1); % true positives
fp = conf_mat(1, 2); % false positives
fn = conf_mat(2, 1); % false negatives
tn = conf_mat(2, 2); % true negatives

precision = tp / (tp + fp);
recall = tp / (tp + fn);
f1_score = 2 * (precision * recall) / (precision + recall);

% Display the results
fprintf('Accuracy: %.2f\n', accuracy);
fprintf('Precision: %.2f\n', precision);
fprintf('Recall: %.2f\n', recall);
fprintf('F1-Score: %.2f\n', f1_score);
