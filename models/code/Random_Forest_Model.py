import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

month_map = {
    'Jan': '1', 'Feb': '2', 'Mar': '3', 'Apr': '4', 'May': '5', 'Jun': '6',
    'Jul': '7', 'Aug': '8', 'Sep': '9', 'Oct': '10', 'Nov': '11', 'Dec': '12'
}

def process_data(spreadsheet_string):
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
    df = df.drop(columns=['Q3', 'Q7'])
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
    df_4 = df_4.apply(convert_word_numbers)
    df_4 = df_4.apply(q4_search_answer_for_numbers)
    df_4_avg = df_4.apply(convert_list_num_to_avg)
    df_4_range = df_4.apply(find_list_range)
    return df_4_avg, df_4_range


# remember to set all lowercase
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
    df['Q8number'] = df['Q8'].copy(deep=True)
    df['Q8number'].replace(["None", "A little (mild)", "A moderate amount (medium)", "A lot (hot)",
                            "I will have some of this food item with my hot sauce"], [0, 1, 2, 3, 4])

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
            # print("error occurred {}".format(date_str))
            pass

    return date_str


def convert_word_numbers(string):
    """
    Finds numbers written out in a string and replaces with a number. E.g. "at least one" -> "at least 1"
    """
    if type(string) is str:
        word_nums = ["two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
                     "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
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
                string = string.replace(word_nums[i], str((i + 3) * 10))

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


def q2boxplots(df):
    lst = ["Pizza", "Shawarma", "Sushi"]
    for i in lst:
        a = df.loc[df['Label'] == i]
    plt.title("Box Plot Showing the Distribution of " + i + "'s # of Ingredients")
    plt.boxplot(a["Q2avg"])


def q4boxplots(df):
    lst = ["Pizza", "Shawarma", "Sushi"]
    for i in lst:
        a = df.loc[df['Label'] == i]
    plt.title("Box Plot Showing the Distribution of " + i + "'s Price Expected to Pay")
    plt.boxplot(a["Q4avg"])



def main():
    print("Loading and preprocessing data...")
    df = process_data("cleaned_data_combined_modified.csv")

    print("Removing rare Q5 movie responses...")
    q5_counts = df['Q5'].value_counts()
    rare_q5_responses = q5_counts[q5_counts <= 1].index.tolist()
    df = df[~df['Q5'].isin(rare_q5_responses)]

    print("Preparing features and labels...")
    non_feature_cols = ['id', 'Q1', 'Q2', 'Q4', 'Q5', 'Q6', 'Q8', 'Label']
    X = df.drop(columns=non_feature_cols, errors='ignore')
    y = df['Label']

    print("Converting object columns to numeric where possible...")
    object_cols = X.select_dtypes(include=['object']).columns
    for col in object_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    X = X.fillna(0)

    print("Splitting into train, validation, and test sets...")
    # First split into train (70%) and temp (30%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    # Then split temp into validation (15%) and test (15%)
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    print("Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=40,
        class_weight='balanced',
        min_samples_split=10,
        min_samples_leaf=1,
        random_state=62
    )
    model.fit(X_train, y_train)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [30, 40, None],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [1, 2]
    }

    grid = GridSearchCV(
        RandomForestClassifier(class_weight='balanced', random_state=42),
        param_grid,
        cv=5,
        scoring='f1_weighted',
        verbose=2
    )
    # grid.fit(X_train, y_train)  # Uncomment to run grid search
    # print("Best Parameters:", grid.best_params_)

    y_pred = model.predict(X_valid)
    print("\nValidation Accuracy: {:.4f}".format(accuracy_score(y_valid, y_pred)))
    print("\nClassification Report:\n", classification_report(y_valid, y_pred))

    importances = model.feature_importances_
    feature_names = X.columns
    sorted_idx = importances.argsort()[::-1]


    # Evaluate on the test set using best hyperparameters
    print("\nTraining best-tuned model for final test evaluation...")
    best_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=40,
        min_samples_split=10,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=42
    )
    best_model.fit(X_train, y_train)

    y_test_pred = best_model.predict(X_test)
    print("\nTest Accuracy: {:.4f}".format(accuracy_score(y_test, y_test_pred)))
    print("\nTest Classification Report:\n", classification_report(y_test, y_test_pred))

    print("\nTop 10 Important Features:")
    for i in range(10):
        print(f"{feature_names[sorted_idx[i]]}: {importances[sorted_idx[i]]:.3f}")

if __name__ == '__main__':
    main()
