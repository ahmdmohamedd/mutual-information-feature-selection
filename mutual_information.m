function mi = mutual_information(X, Y)
    % Mutual Information between continuous X and discrete Y
    % X: continuous feature
    % Y: discrete labels

    % Ensure Y is a column vector
    Y = Y(:);
    
    % Calculate the histogram for the continuous feature
    [counts, edges] = histcounts(X, 'Normalization', 'probability');
    
    % Calculate the probabilities of Y
    unique_Y = unique(Y);
    pY = histcounts(Y, 'Normalization', 'probability');
    
    % Calculate the conditional probabilities P(X|Y)
    pX_given_Y = zeros(length(unique_Y), length(counts));
    
    for k = 1:length(unique_Y)
        idx = (Y == unique_Y(k));
        pX_given_Y(k, :) = histcounts(X(idx), edges, 'Normalization', 'probability');
    end
    
    % Calculate mutual information
    mi = 0;
    for k = 1:length(unique_Y)
        if pY(k) > 0
            % Avoid division by zero and log(0)
            valid_probs = pX_given_Y(k, :) > 0;
            mi = mi + pY(k) * sum(pX_given_Y(k, valid_probs) .* log2(pX_given_Y(k, valid_probs) ./ counts(valid_probs)));
        end
    end
end
