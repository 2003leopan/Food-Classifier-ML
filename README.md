Machine Learning - Food-Item Classifier

Built an end-to-end text-and-categorical ML pipeline from a 1,644-response survey; expanded 8 raw questions into 435 engineered features (keyword indicators, checkbox one-hots, numeric/range parsers, simple FX normalization for prices, robust NaN handling).

Compared Neural Network, Random Forest, and Multinomial Naive Bayes; conducted systematic hyperparameter search (GridSearchCV for RF; multi-grid sweeps for NN with 10× repeats).

Final model: Neural Network (2 hidden layers, ReLU/Softmax, lr=0.01, batch=64, 200 epochs) with 0.916 validation accuracy; test accuracy point estimate 89.82% based on 1,000 randomized splits; RF benchmarked at 87.7%.

Identified high-signal features (e.g., Food Setting, “Who it reminds you of,” hot-sauce preference) and pruned sparse one-offs to improve generalization.

Ensured reproducibility (seeded runs, vectorized ops), clean train/val/test separation (70/15/15), and no leakage.

Role: Implemented Random Forest & tuning; co-designed feature pipeline; contributed to model selection & report.
Tech: Python (NumPy/pandas, scikit-learn), basic NN implementation, data viz (heatmaps/boxplots).

Achieved 2nd place in final test accuracy amongst teams in all campuses at Uuniversity of Toronto.
