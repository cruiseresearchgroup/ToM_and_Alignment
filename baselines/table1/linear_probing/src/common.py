from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import warnings


def fit_logistic_regression(X, y):
    """
    Fits a logistic regression model using a pipeline of scaling and classification.

    Parameters:
    - X: numpy array of shape [Num_samples, Num_features]
    - y: numpy array of shape [Num_samples]

    Returns:
    - pipeline: A Pipeline object with the fitted StandardScaler and LogisticRegression model
    """
    # Create a pipeline that first scales the data and then applies logistic regression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000))  # Increase max_iter if needed
    ])

    # Fit the pipeline to the entire dataset
    pipeline.fit(X, y)

    return pipeline

def fit_ridge_regression_pipeline(X, y, X_val=None, y_val=None, param_grid=None, random_state=29):
    warnings.filterwarnings('ignore')
    
    pipeline = Pipeline([
        ('pca', PCA(random_state=random_state)),
        ('regressor', Ridge(random_state=random_state))
    ])
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    grid_search.fit(X, y)
    
    return grid_search.best_estimator_, grid_search.best_score_, grid_search.best_params_



def fit_logistic_regression_with_pca(X, y, n_components=100):
    """
    Fits a logistic regression model using a pipeline of PCA, scaling, and classification.

    Parameters:
    - X: numpy array of shape [Num_samples, Num_features]
    - y: numpy array of shape [Num_samples]
    - n_components: Number of components to keep for PCA. If None, all components are kept.

    Returns:
    - pipeline: A Pipeline object with the fitted PCA, StandardScaler, and LogisticRegression model
    """
    # Create a pipeline that first applies PCA, then scales the data, and finally applies logistic regression
    pipeline = Pipeline([
        ('pca', PCA(n_components=n_components)),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000))  # Increase max_iter if needed
    ])

    # Fit the pipeline to the entire dataset
    pipeline.fit(X, y)

    return pipeline



def train_and_tune_logistic_regression(X_train, y_train, X_valid, y_valid, param_grid):
    """
    Trains and tunes a logistic regression model using a pipeline, performs hyperparameter tuning, 
    and evaluates on a validation set.

    Parameters:
    - X_train: numpy array of shape [Num_samples_train, Num_features] (training features)
    - y_train: numpy array of shape [Num_samples_train] (training labels)
    - X_valid: numpy array of shape [Num_samples_valid, Num_features] (validation features)
    - y_valid: numpy array of shape [Num_samples_valid] (validation labels)
    - param_grid: dictionary defining hyperparameter grid for tuning

    Returns:
    - best_pipeline: Trained pipeline with the best parameters
    - validation_metrics: Dictionary with validation accuracy, F1 score, and classification report
    """
    print('gird search started')
    # Create the pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000))  # Logistic Regression with increased iterations
    ])

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, scoring='f1_weighted', cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best pipeline
    best_pipeline = grid_search.best_estimator_

    # Evaluate on the validation set
    y_valid_pred = best_pipeline.predict(X_valid)
    validation_metrics = {
        'accuracy': accuracy_score(y_valid, y_valid_pred),
        'f1_score': f1_score(y_valid, y_valid_pred, average='weighted'),
        'classification_report': classification_report(y_valid, y_valid_pred)
    }

    # Print validation metrics
    print("Validation Metrics:")
    print(f"Accuracy: {validation_metrics['accuracy']:.4f}")
    print(f"F1 Score: {validation_metrics['f1_score']:.4f}")
    print(validation_metrics['classification_report'])

    return best_pipeline, validation_metrics


def plot_nested_dict(data, xlabel='X-axis', ylabel='Y-axis', title='Line Plots'):
    """
    Plots line plots from a dictionary of dictionaries.

    Parameters:
    - data: dict of dicts
        Outer dictionary keys are used for the legend.
        Inner dictionary keys are the x-values, and values are the y-values.
    - xlabel: str, label for the X-axis (default: 'X-axis')
    - ylabel: str, label for the Y-axis (default: 'Y-axis')
    - title: str, title of the plot (default: 'Line Plots')
    """
    plt.figure(figsize=(10, 6))

    for outer_key, inner_dict in data.items():
        # Ensure the inner dictionary keys are sorted for consistent plotting
        x_values = sorted(inner_dict.keys())
        y_values = [inner_dict[x] for x in x_values]
        plt.plot(x_values, y_values, label=str(outer_key))  # Convert key to string for the legend

    # Add plot labels and legend
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(title='Legend', loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Show the plot
    plt.show()

def boxplot(signals, train=[4, 5, 6], test=[4, 5, 6], figsize=(12, 7)):

    # Set Seaborn style
    sns.set(style="whitegrid", palette="muted", font_scale=1.2)

    fig, axes = plt.subplots(len(train), len(test), figsize=figsize, sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    probs_name = list(set([item.split('_')[-1] for item in signals.keys()]))
    idx = 0
    
    for i in train:
        train_length = i
        for j in test:
            test_length = j
            ax = axes.flat[idx]
            boxplot_data = []
            labels = []
            for prb in probs_name:
                key = f'{train_length}_{test_length}_{prb}'
                value = list(signals[key].values())
                boxplot_data.append(value)
                labels.append(prb)

            # Use seaborn's boxplot for a fancier look
            sns.boxplot(data=boxplot_data, ax=ax, palette="Set2", showcaps=False, boxprops=dict(alpha=0.6))
            
            # Adjust labels and title
            ax.set_xticklabels(labels, rotation=90)
            ax.set_title(f'Train Length: {train_length} | Test Length: {test_length}', fontsize=10)
            
            # Add grid and some style
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.set_xlabel('Preferences', fontsize=9)
            ax.set_ylabel("Prob's Accuracy", fontsize=9)
            
            idx += 1
    
    plt.tight_layout()
    plt.show()



def lineplot(signals, train=[4, 5, 6], test=[4, 5, 6], figsize=(12, 7)):

    fig, axes = plt.subplots(len(train), len(test), figsize=figsize, sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.2, wspace=0.1)
    probs_name = list(set([item.split('_')[-1] for item in signals.keys()]))
    idx = 0
    
    for i in train:
        train_length = i
        for j in test:
            test_length = j
            ax = axes.flat[idx]
            # Prepare data for lineplot
            for prb in probs_name:
                key = f'{train_length}_{test_length}_{prb}'
                value = list(signals[key].values())
                
                # Create a lineplot for the current probability
                sns.lineplot(x=list(range(len(value))), y=value, ax=ax, label=prb, marker='o', legend=False)
                
            ax.set_title(f'Train Length:{train_length}|Test Length:{test_length}', fontsize=8)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.set_xlabel('Layers', fontsize=9)
            ax.set_ylabel("Prob's Accuracy", fontsize=9)
            idx += 1

    # Create a single legend for all subplots
    handles, labels = axes[0, 0].get_legend_handles_labels()  # Get handles and labels from one of the subplots
    fig.legend(handles, labels, loc='upper center', ncol=len(probs_name), fontsize=10)
    
    plt.show()