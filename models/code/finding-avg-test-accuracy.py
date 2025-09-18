import re
import pandas as pd
import numpy as np

month_map = {
    'Jan': '1',
    'Feb': '2',
    'Mar': '3',
    'Apr': '4',
    'May': '5',
    'Jun': '6',
    'Jul': '7',
    'Aug': '8',
    'Sep': '9',
    'Oct': '10',
    'Nov': '11',
    'Dec': '12'
}

#Note: cast q1 values from int to str? or unnecessary 
def process_data(spreadsheet_string):
    """
    returns a dataframe that reinterprets answers in a usable form
    """
    df = pd.read_csv(spreadsheet_string)
    df.columns = ['id', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Label']
    df['Q2avg'], df['Q2range'] = process_q2(df['Q2'])
    df['Q4avg'], df['Q4range'] = process_q4(df['Q4'])
    df = process_q2_ingredients(df)
    df = at_least_or_more(df)
    df = process_q3(df)
    df = process_q5(df)
    df = process_q6(df)
    df = process_q7(df)
    df = process_q8(df)
    df = df.drop(columns=['Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8'])
    
    return df

def process_q2(df_2):
    """
    Takes the column for the responses for Q2.
    Returns two dataframe columns: interpreting the average ingredient, the
    range of ingredients (e.g. 2 if the response is 3-5)
    """
    df_2 = df_2.apply(convert_date_format)
    df_2 = df_2.apply(to_lowercase)
    df_2 = df_2.apply(convert_word_numbers)
    df_2 = df_2.apply(q2_search_answer_for_numbers)
    df_2 = df_2.replace(np.nan, 0)
    df_2_avg = df_2.apply(convert_list_num_to_avg)
    df_2_range = df_2.apply(find_list_range)
    
    return df_2_avg, df_2_range

def process_q3(df):
    """
    Creates indicator variables for each setting in Q3 (e.g. Weekday Lunch)
    """
    lst = ['Week day lunch', 'Week day dinner', 'Weekend lunch', 'Weekend dinner', 'At a party', 'Late night snack']
    for i in lst:
        df[i] = df['Q3'].str.contains(i)
        df[i] = df[i].replace(np.nan, False)
        df[i] = df[i].astype(int)

    return df

def process_q4(df_4):
    """
    Takes the column for the responses for Q4.
    Returns two dataframe columns: interpreting the average cost, the
    range of cost
    """
    df_4 = df_4.apply(convert_date_format)
    df_4 = df_4.apply(to_lowercase)
    df_4 = df_4.replace(np.nan, 0)
    df_4 = df_4.apply(convert_word_numbers)
    df_4 = df_4.apply(q4_search_answer_for_numbers)
    df_4_avg = df_4.apply(convert_list_num_to_avg)
    df_4_range = df_4.apply(find_list_range)
    
    return df_4_avg, df_4_range

#remember to set all lowercase
def process_q5(df):
    """
    Appends new columns to dataframe to indicate (1 or 0) if a response's Q5 
    answer contains a key movie word.
    """
    m = pd.read_excel("CSC311 Keywords.xlsx", sheet_name="Q5 Movie")
    movie_keywords = m['Keyword'].to_list()
    df['Q5'] = df['Q5'].apply(to_lowercase)
    for i in range(len(movie_keywords)):
        temp = df['Q5'].apply(search, key=movie_keywords[i])
        temp.name = movie_keywords[i]
        df = pd.concat([df, temp], axis=1)
        
    return df

def process_q6(df):
    """
    Appends new columns to dataframe to indicate (1 or 0) if a response's Q6 
    answer contains a key drink word.
    """
    d = pd.read_excel("CSC311 Keywords.xlsx", sheet_name="Q6 Drink")
    drinks = d['Drink'].to_list()
    drinks = [str(x) for x in drinks]
    df['Q6'] = df['Q6'].apply(to_lowercase)

    for i in range(len(drinks)):
        temp = df['Q6'].apply(search, key=drinks[i])
        temp.name = drinks[i]
        df = pd.concat([df, temp], axis=1)

    return df

def process_q7(df):
    """
    Creates indicator variables for each people group in Q7 (who do you associate food item with)
    """
    lst = ['Parents', 'Siblings', 'Strangers', 'Teachers', 'Friends']
    for i in lst:
        df[i] = df['Q7'].str.contains(i)
        df[i] = df[i].replace(np.nan, False)
        df[i] = df[i].astype(int)

    return df

def process_q8(df):
    """
    Creates a numerical variable converting Q8 (amt of hot sauce) responses to a 1-5 scale
    Precondition: dataframe contains a column 'Q8'
    """
    # Create a mapping dictionary for hot sauce preferences
    hot_sauce_map = {
        "None": 0,
        "A little (mild)": 1,
        "A moderate amount (medium)": 2,
        "A lot (hot)": 3,
        "I will have some of this food item with my hot sauce": 4
    }
    
    # Create a new column with numeric values
    df['Q8number'] = df['Q8'].map(hot_sauce_map)
    
    # Fill any NaN values with 0 (assuming no preference means no hot sauce)
    df['Q8number'] = df['Q8number'].fillna(0)
    
    # Ensure the column is numeric
    df['Q8number'] = pd.to_numeric(df['Q8number'], errors='coerce').fillna(0)

    return df

def process_q2_ingredients(df):
    """
    Appends new columns to dataframe to indicate (1 or 0) if a response's Q2 
    answer contains a key ingredient.
    """
    i = pd.read_excel("CSC311 Keywords.xlsx", sheet_name="Q2 Ingredients named")
    ingredient = i['Ingredient'].to_list()
    
    for i in range(len(ingredient)):
        temp = df['Q2'].apply(search, key=ingredient[i])
        temp.name = ingredient[i]
        df = pd.concat([df, temp], axis=1)
        
    return df

def at_least_or_more(df):
    """
    Given responses in Q2, create indicator two indicator columns: first
    if a minimum was indicated (e.g. 3+, at least 3, or more, more than, minimum)
    or a maximum (at most, maximum, less than)
    "Dumb" system since it doesn't track the number associated with the max/min
    """
    min_keywords = ['+', 'at least', 'or more', 'minimum', 'more than']
    max_keywords = ['at most', 'or less', 'or fewer', 'fewer than', 'less than', 'maximum']
    df['min_ingredient'] = df['Q2'].apply(search_list, keywords=min_keywords)
    df['max_ingredient'] = df['Q2'].apply(search_list, keywords=max_keywords)
    
    return df

def search_list(string, keywords):
    """
    Takes in a string and returns 1 if any of the keywords are present
    returns 0 otherwise.
    """
    
    for i in keywords:
        if i in string:
            return 1
    return 0

def search(string, key):
    """
    Takes a string. Returns 1 if key is found in string, 0 otherwise.
    """
    if type(string) is str and key in string:
        return 1
    else:
        return 0

# Function to convert date format from Prof. Gao
def convert_date_format(date_str):
    if isinstance(date_str, str) and '-' in date_str:
        try:
            day, month_abbr = date_str.split('-')
            month_num = month_map.get(month_abbr, month_abbr)
            return f"{int(month_num)}-{int(day)}"
        except ValueError:
           #print("error occurred {}".format(date_str))
           pass

    return date_str

def convert_word_numbers(string):
    """
    Finds numbers written out in a string and replaces with a number. E.g. "at least one" -> "at least 1"
    """
    if type(string) is str:
        word_nums = ["two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
        large_word_nums = ["thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred"]
        
        for i in range(len(word_nums)):
            result = string.find(word_nums[i])
            check = string.find(word_nums[i] + "t")
            if i < 9 and check > -1 and result == check:
                continue
            if result > -1:
                string = string.replace(word_nums[i], str(i + 2))
                
        for i in range(len(large_word_nums)):
            if string.find(large_word_nums[i]) > -1:
                string = string.replace(word_nums[i], str((i + 3)*10))
        
    return string

def q2_search_answer_for_numbers(string):
    """
    Given a response, return a list of integers containing all the numbers contained.
    SPECIAL CASE: response includes the word "total", so we take only max value.
    E.g. "6to7" -> [6, 7]
    """
    list_num = re.findall(r'\d+', string)
    list_num = [int(x) for x in list_num]

    if len(list_num) == 0:
        num_commas = string.count(',')
        if num_commas > 1:
            list_num.append(num_commas + 1)
        else:
            list_num.append(len(string.split()))
        
    if string.find("total") > -1:
        list_num = [max(list_num)]
        
    return list_num

def q4_search_answer_for_numbers(string):
    """
    Given a response, return a list of integers containing all the numbers contained.
    SPECIAL CASE: response includes the word "total", so we take only max value.
    E.g. "6to7" -> [6, 7]
    """
    if type(string) is str:
        list_num = re.findall(r'\d+[\.]*\d*', string)
        list_num = [float(x) for x in list_num]
        if "couple" in string and list_num == []:
            list_num.append(2)
        elif "few" in string and list_num == []:
            list_num.append(3)
        elif "several" in string and list_num == []:
            list_num.append(4)
    else:
        list_num = [string]
        
    return list_num

def convert_list_num_to_avg(list_num):
    """
    Takes a list of integers.
    Returns the average of the greatest 2 numbers, or -1 if the list is empty.
    Returns the sole number if the length of the list is 1.
    """
    if len(list_num) == 0:
        return -1
    elif len(list_num) == 1:
        return list_num[0]
    elif len(list_num) == 2:
        return (list_num[0] + list_num[1]) / 2
    else:
        list_num.sort()
        return (list_num[-1] + list_num[-2]) / 2

def find_list_range(list_num):
    """
    Takes a list of integers.
    Returns the difference between the 2 largest numbers is the list, 0 otherwise.
    """
    
    if len(list_num) < 2:
        return 0
    elif len(list_num) == 2:
        if list_num[0] > list_num[1]:
            return list_num[0]
        return abs(list_num[1] - list_num[0])
    else:
        list_num.sort()
        return abs(list_num[-2] - list_num[-1])

def to_lowercase(string):
    if type(string) is str:
        return string.lower()
    
def make_onehot(indicies, total):
    """
    Convert indicies into one-hot vectors.
    
    Parameters:
        `indices` - numpy array of indices
        `total` - total number of categories (3 in our case)
    
    Returns: numpy array of one-hot vectors
    """
    I = np.eye(total)
    return I[indicies]
    
def prepare_data(df):
    """
    Prepare the data for neural network training by selecting relevant features
    and converting labels to one-hot encoding
    """
    # Select features for training (excluding 'id' and 'Label')
    feature_columns = [col for col in df.columns if col not in ['id', 'Label', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8']]
    
    # Fill NaN values with appropriate values based on column type
    for col in feature_columns:
        col_type = str(df[col].dtypes)
        if col_type in ['int64', 'float64']:
            # For numeric columns, fill with median
            df[col] = df[col].fillna(df[col].median())
        else:
            # For categorical columns, fill with mode
            mode_values = df[col].mode()
            if not mode_values.empty:
                df[col] = df[col].fillna(mode_values.iloc[0])
            else:
                # If no mode exists, fill with 0 for binary features
                df[col] = df[col].fillna(0)
    
    X = df[feature_columns].values
    print("\nFeature matrix shape:", X.shape)
    
    # Ensure all values are float
    X = X.astype(float)
    
    # Normalize features
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1  # Prevent division by zero
    X = (X - X_mean) / X_std
    
    # Convert labels to one-hot encoding
    unique_labels = df['Label'].unique()
    print("\nUnique labels:", unique_labels)
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    print("Label to index mapping:", label_to_idx)
    
    y = np.array([label_to_idx[label] for label in df['Label']])
    
    y_onehot = make_onehot(y, total=len(unique_labels))
    
    return X, y_onehot, label_to_idx

def split_data(X, y, train_ratio=0.8, val_ratio=0.2, test_ratio=0.2):
    """
    Split data into training, validation, and test sets
    """
    print("Input X shape:", X.shape)
    print("Input y shape:", y.shape)
    
    n_samples = len(X)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val
    
    print(f"\nTotal samples: {n_samples}")
    print(f"Training samples: {n_train}")
    print(f"Validation samples: {n_val}")
    print(f"Test samples: {n_test}")
    
    # Shuffle indices
    indices = np.random.permutation(n_samples)
    
    # Split indices
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    test_indices = indices[n_train+n_val:]
    
    # Split data
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def softmax(z):
    """
    Compute the softmax of vector z, or row-wise for a matrix z.
    For numerical stability, subtract the maximum logit value from each
    row prior to exponentiation
    """
    m = np.max(z, axis=1, keepdims=True)
    return np.exp(z - m) / np.sum(np.exp(z - m), axis=1, keepdims=True)

def do_forward_pass(model, X):
    """
    Compute the forward pass of the MLP.
    """
    model.N = X.shape[0]
    model.X = X
    model.m = np.dot(X, model.W1.T) + model.b1
    model.h = np.maximum(0, model.m) 
    model.z = np.dot(model.h, model.W2.T) + model.b2
    model.y = softmax(model.z)
    return model.y
    
    return model.y

def do_backward_pass(model, ts):
    """
    Compute the backward pass, given the ground-truth, one-hot targets.

    You may assume that `model.forward()` has been called for the
    corresponding input `X`, so that the quantities computed in the
    `forward()` method is accessible.

    Parameters:
        `model` - An instance of the class MLPModel
        `ts` - A numpy array of shape (N, model.num_classes)
    """
    model.z_bar = (model.y - ts) / model.N
    model.W2_bar = np.dot(model.z_bar.T, model.h)
    model.b2_bar = np.sum(model.z_bar, axis=0)
    model.h_bar = np.dot(model.z_bar, model.W2)
    model.m_bar = model.h_bar * (model.m > 0)
    model.W1_bar = np.dot(model.m_bar.T, model.X)
    model.b1_bar = np.sum(model.m_bar, axis=0)
    
    return model.W1_bar, model.b1_bar, model.W2_bar, model.b2_bar

class MLPModel(object):
    def __init__(self, num_features, num_hidden=100, num_classes=3):
        """
        Initialize the weights and biases of this two-layer MLP.
        """
        # information about the model architecture
        self.num_features = num_features
        self.num_hidden = num_hidden
        self.num_classes = num_classes

        # weights and biases for the first layer of the MLP
        self.W1 = np.zeros([num_hidden, num_features])
        self.b1 = np.zeros([num_hidden])

        # weights and biases for the second layer of the MLP
        self.W2 = np.zeros([num_classes, num_hidden])
        self.b2 = np.zeros([num_classes])

        # initialize the weights and biases
        self.initializeParams()

        # set all values of intermediate variables to None
        self.cleanup()

    def initializeParams(self):
        """
        Initialize the weights and biases of this two-layer MLP to be random.
        This random initialization is necessary to break the symmetry in the
        gradient descent update for our hidden weights and biases. 
        """
        # Use smaller initialization values
        self.W1 = np.random.normal(0, 2/self.num_features, self.W1.shape)
        self.b1 = np.random.normal(0, 2/self.num_features, self.b1.shape)
        self.W2 = np.random.normal(0, 2/self.num_hidden, self.W2.shape)
        self.b2 = np.random.normal(0, 2/self.num_hidden, self.b2.shape)

    def forward(self, X):
        """
        Compute the forward pass to produce prediction for inputs.

        Parameters:
            `X` - A numpy array of shape (N, self.num_features)

        Returns: A numpy array of predictions of shape (N, self.num_classes)
        """
        return do_forward_pass(self, X)

    def backward(self, ts):
        """
        Compute the backward pass, given the ground-truth, one-hot targets.

        You may assume that the `forward()` method has been called for the
        corresponding input `X`, so that the quantities computed in the
        `forward()` method is accessible.

        Parameters:
            `ts` - A numpy array of shape (N, self.num_classes)
        """
        return do_backward_pass(self, ts)

    def loss(self, ts):
        """
        Compute the average cross-entropy loss, given the ground-truth, one-hot targets.

        You may assume that the `forward()` method has been called for the
        corresponding input `X`, so that the quantities computed in the
        `forward()` method is accessible.

        Parameters:
            `ts` - A numpy array of shape (N, self.num_classes)
        """
        return np.sum(-ts * np.log(self.y)) / ts.shape[0]

    def update(self, alpha):
        """
        Compute the gradient descent update for the parameters.
        """
        self.W1 = self.W1 - alpha * self.W1_bar
        self.b1 = self.b1 - alpha * self.b1_bar
        self.W2 = self.W2 - alpha * self.W2_bar
        self.b2 = self.b2 - alpha * self.b2_bar

    def cleanup(self):
        """
        Erase the values of the variables that we use in our computation.
        """
        self.N = None
        self.X = None
        self.m = None
        self.h = None
        self.z = None
        self.y = None
        self.z_bar = None
        self.W2_bar = None
        self.b2_bar = None
        self.h_bar = None
        self.m_bar = None
        self.W1_bar = None
        self.b1_bar = None

def train_sgd(model, X_train, t_train,
              alpha=0.001, n_epochs=0, batch_size=32,
              X_valid=None, t_valid=None,
              X_test=None, t_test=None,
              w_init=None, plot=True):
    '''
    Given `model` - an instance of MLPModel
          `X_train` - the data matrix to use for training
          `t_train` - the target vector to use for training
          `alpha` - the learning rate.
                    From our experiments, it appears that a larger learning rate
                    is appropriate for this task.
          `n_epochs` - the number of **epochs** of gradient descent to run
          `batch_size` - the size of each mini batch
          `X_valid` - the data matrix to use for validation (optional)
          `t_valid` - the target vector to use for validation (optional)
          `X_test` - the data matrix to use for testing (optional)
          `t_test` - the target vector to use for testing (optional)
          `w_init` - the initial `w` vector (if `None`, use a vector of all zeros)
          `plot` - whether to track statistics and plot the training curve

    Solves for model weights via stochastic gradient descent,
    using the provided batch_size.

    Return weights after `niter` iterations.
    '''
    # as before, initialize all the weights to zeros
    w = np.zeros(X_train.shape[1])

    train_loss = [] # for the current minibatch, tracked once per iteration
    valid_loss = [] # for the entire validation data set, tracked once per epoch

    # track the number of iterations
    niter = 0

    # we will use these indices to help shuffle X_train
    N = X_train.shape[0] # number of training data points
    indices = list(range(N))

    for e in range(n_epochs):
        np.random.shuffle(indices) # for creating new minibatches

        for i in range(0, N, batch_size):
            if (i + batch_size) > N:
                # At the very end of an epoch, if there are not enough
                # data points to form an entire batch, then skip this batch
                continue

            indices_in_batch = indices[i: i+batch_size]
            X_minibatch = X_train[indices_in_batch, :]
            t_minibatch = t_train[indices_in_batch]  # Already one-hot encoded

            # gradient descent iteration
            model.cleanup()
            model.forward(X_minibatch)
            model.backward(t_minibatch)
            model.update(alpha)

            if plot:
                # Record the current training loss values
                train_loss.append(model.loss(t_minibatch))
            niter += 1

        # compute validation data metrics, if provided, once per epoch
        if plot and (X_valid is not None) and (t_valid is not None):
            model.cleanup()
            model.forward(X_valid)
            valid_loss.append((niter, model.loss(t_valid)))
            
            # Print epoch progress and prediction distribution
            train_preds = np.argmax(model.forward(X_train), axis=1)
            val_preds = np.argmax(model.forward(X_valid), axis=1)
            print(f"\nEpoch {e+1}/{n_epochs}")
            print(f"Training Loss: {train_loss[-1]:.4f}, Validation Loss: {valid_loss[-1][1]:.4f}")

    if plot:
        print("\nTraining Complete!")
        print("Final Training Loss:", train_loss[-1])
        if (X_valid is not None) and (t_valid is not None):
            print("Final Validation Loss:", valid_loss[-1][1])
            
            # Calculate final accuracies
            model.forward(X_train)
            train_preds = np.argmax(model.y, axis=1)
            train_true = np.argmax(t_train, axis=1)
            train_accuracy = np.mean(train_preds == train_true)
            
            model.forward(X_valid)
            val_preds = np.argmax(model.y, axis=1)
            val_true = np.argmax(t_valid, axis=1)
            val_accuracy = np.mean(val_preds == val_true)
            
            print("Final Training Accuracy:", train_accuracy)
            print("Final Validation Accuracy:", val_accuracy)


def main():
    # Load and process data
    print("\nLoading data...")
    df = process_data("cleaned_data_combined_modified.csv")
    print("\nData loaded successfully!")
    
    # Prepare data for neural network
    X, y, label_to_idx = prepare_data(df)
    
    # Initialize list to store accuracies
    training_accuracies = []
    validation_accuracies = []
    test_accuracies = []
    
    # Run 1000 times
    num_runs = 1000
    print(f"\nRunning {num_runs} times with the same parameters...")
    
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        
        # Split data into train, validation, and test sets
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X, y, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
                
        # Initialize model
        model = MLPModel(X.shape[1], 
                        num_hidden=200,
                        num_classes=y.shape[1])
        
        # Train model
        train_sgd(model, X_train, y_train,
                    alpha=0.01,
                    n_epochs=200,
                    batch_size=64,
                    X_valid=X_val, t_valid=y_val,
                    X_test=X_test, t_test=y_test,
                    plot=False)  # Set plot to False to reduce output
        
        # Calculate accuracies
        model.forward(X_train)
        train_preds = np.argmax(model.y, axis=1)
        train_true = np.argmax(y_train, axis=1)
        train_accuracy = np.mean(train_preds == train_true)
        training_accuracies.append(train_accuracy)
        
        model.forward(X_val)
        val_preds = np.argmax(model.y, axis=1)
        val_true = np.argmax(y_val, axis=1)
        val_accuracy = np.mean(val_preds == val_true)
        validation_accuracies.append(val_accuracy)
        
        model.forward(X_test)
        test_preds = np.argmax(model.y, axis=1)
        test_true = np.argmax(y_test, axis=1)
        test_accuracy = np.mean(test_preds == test_true)
        test_accuracies.append(test_accuracy)
    
    # Calculate final statistics
    avg_train_acc = np.mean(training_accuracies)
    std_train_acc = np.std(training_accuracies)
    avg_val_acc = np.mean(validation_accuracies)
    std_val_acc = np.std(validation_accuracies)
    avg_test_acc = np.mean(test_accuracies)
    std_test_acc = np.std(test_accuracies)
    
    print("\nFinal Results after 1000 runs:")
    print(f"Average Training Accuracy: {avg_train_acc:.4f} ± {std_train_acc:.4f}")
    print(f"Average Validation Accuracy: {avg_val_acc:.4f} ± {std_val_acc:.4f}")
    print(f"Average Test Accuracy: {avg_test_acc:.4f} ± {std_test_acc:.4f}")
    
    # Print best and worst runs
    best_train_idx = np.argmax(training_accuracies)
    worst_train_idx = np.argmin(training_accuracies)
    print(f"\nBest Training Accuracy: {training_accuracies[best_train_idx]:.4f} (Run {best_train_idx + 1})")
    print(f"Worst Training Accuracy: {training_accuracies[worst_train_idx]:.4f} (Run {worst_train_idx + 1})")



if __name__ == '__main__':
    main()
