####################################################################################################
# It is requested to develop a machine learning model that can predict
# whether individuals have diabetes or not when their features are specified.
# Before developing the model, it is expected that you perform the necessary
# data analysis and feature engineering steps.
####################################################################################################

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

##################################################
# The accuracy without feature engineering process
##################################################

dff = pd.read_csv("Dataset/diabetes.csv")
dff.dropna(inplace=True)

y = dff["Outcome"]
X = dff.drop(["Outcome"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X_train)

# Accuracy score is 0.77 and most important feature is Glucose

##################################################
# Exploratory Data Analysis
##################################################

df_= pd.read_csv("Dataset/diabetes.csv")
df = df_.copy()


def check_df(dataframe):
    print("##### Head of the dataframe ##### \n")
    print(dataframe.head(), "\n")
    print("##### Shape of the dataframe ##### \n")
    print(dataframe.shape, "\n")
    print("##### Infomartion about variables ##### \n")
    print(dataframe.info(), "\n")
    print("##### Descriptive statistics of dataframe ##### \n")
    print(dataframe.describe().T, "\n")

check_df(df)

# Grabbing numerical and categorical variable names

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    The dataset provides the names of categorical, numeric, and categorical but cardinal variables. Note that numeric-looking categorical variables are also included in the categorical variables.

    Parameters
    ------
        dataframe: dataframe
                The dataframe from which variable names are desired to be taken
        cat_th: int, optional
                Class threshold value for variables that are numeric but categorical
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                List of categorical variables
        num_cols: list
                List of numerical variables
        cat_but_car: list
                List of categorical-looking cardinal variables

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        cat_cols includes num_bat_cat
        Total of returned 3 list equal to the total number of variables:
        cat_cols + num_cols + cat_but_car = num of variables

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Only categorical variable is "Outcome" which is our target variable.

## Analysis of categorical and numerical variables ##

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

cat_summary(df, "Outcome", plot=True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot = True)

# We can observe an accumulation around 0 on the count plot for certain variables, such as Glucose and Insulin.

## Analysis of target variable ##

df.groupby(cat_cols)[num_cols].mean()

# We can observe a significant difference between diabetes and non-diabetes patients in certain variables such as Insulin and Glucose.
# However, it would not be appropriate to make a statistical inference solely based on the mean values obtained from these variables.


# Analysis of outlier values #

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(df, variable):
    low_limit, up_limit = outlier_thresholds(df, variable)
    if df.loc[(df[variable] < low_limit) | (df[variable]  > up_limit)].any(axis=None):
         return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col))

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

grab_outliers(df, "Insulin") # Outliers in "Insulin" variable
len(grab_outliers(df, "Insulin", True)) # Number of outliers

# Since we have 768 observations, it would be better to use the IQR method to adjust the boxplot and remove outlier values.

def replace_with_thresholds(df, variable):
    low_limit, up_limit = outlier_thresholds(df, variable)
    df.loc[(df[variable] < low_limit), variable] = low_limit
    df.loc[(df[variable] > up_limit), variable] = up_limit


for cols in num_cols:
    replace_with_thresholds(df, cols)

# Check again for outliers

for col in num_cols:
    print(col, check_outlier(df, col))

# LOF Analysis

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)
df_scores = clf.negative_outlier_factor_

np.sort(df_scores)[0:5]

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show()

# We take the first breaking point of elbow, which is fourth index
th = np.sort(df_scores)[4]

# We analyse the results and inspect outliers' statistical features
df[df_scores < th]
df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

# High Glucose values were found with lower BMI and BloodPressure, which may indicate a singularity issue.

# Update dataframe without outliers

df = df.drop(axis=0, labels=df[df_scores < th].index)


# Analysis of missing values #

df.isnull().sum()
df.isnull().values.any()

# No missing values were found in the data. Any missing values were filled with 0, as determined during the analysis of variables.
# Therefore, we will only need to identify variables with unnatural 0 values for future analysis.

for col in num_cols:
    num_summary(df, col, plot = True)

# SkinThickness and Insulin variables have 0 values while they shouldn't. Other numerical variables
# such as Pregnancies, BMI and Age look normal because their min value already bigger than 0.

nan_cols = ["Insulin", "SkinThickness"]


# Correlation analysis

df.corr().sort_values("Outcome", ascending=False).drop("Outcome", axis=0)

sns.heatmap(df.corr(), annot=True)
plt.show()

# We can see positive correlation between Insulin-SkinThickness, Age-SkinThickness
# and Glucose-Outcome. While these correlations are moderate, they should still be considered in the next steps.

##################################################
# Feature Engineering
##################################################

# There are no missing observations in the dataset, but observation units containing a value of 0
# for variables such as Glucose, Insulin, etc. may represent missing values.
# For example, a person's glucose or insulin value cannot be 0.
# Considering this situation, we assign zero values to NaN in the relevant variables and then apply operations to missing values.

# Replacing 0 values with NaN
for col in nan_cols:
    df[col] = df[col].replace(0, np.nan)

# Checking missing values ratios and counts in variables
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df, True)

# Inspecting missing values and relation between them
msno.bar(df)
msno.matrix(df)
msno.heatmap(df) # There is a positive correlation between missing values of Insulin and SkinThickness
plt.show()


## Handling Missing Data ##
# After trying out all three methods on the data, it was found that using median values yielded the best result.

# Imputation with KNN

dft = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)

dft.head()

scaler = MinMaxScaler()
dft = pd.DataFrame(scaler.fit_transform(dft), columns=dft.columns)
dft.head()

# KNN
imputer = KNNImputer(n_neighbors=5)
dft = pd.DataFrame(imputer.fit_transform(dft), columns=dft.columns)
dft.head()

dft = pd.DataFrame(scaler.inverse_transform(dft), columns=dft.columns)

df["Insulin"] = dff[["Insulin"]]
df["SkinThickness"] = dff[["SkinThickness"]]
df.isnull().sum()


# Imputation with mean values

df["SkinThickness"].fillna(df.groupby("Outcome")["SkinThickness"].transform("mean"), inplace=True)
df["Insulin"].fillna(df.groupby('Outcome')["Insulin"].transform("mean"), inplace=True)
df.isnull().sum()

# Imputation with median values

df["SkinThickness"].fillna(df.groupby("Outcome")["SkinThickness"].transform("median"), inplace=True)
df["Insulin"].fillna(df.groupby('Outcome')["Insulin"].transform("median"), inplace=True)
df.isnull().sum()

## Feature Extraction ##

df["age_period"] = pd.cut(df["Age"],
                         bins = [df.Age.min(), 33,
                         45, df.Age.max()],
                         labels = ["Young", "Adult", "Mature"],
                         include_lowest=True)

def calculate_bmi_range(dataframe):
    if dataframe["BMI"] < 18.5:
        return "underweight"
    elif dataframe["BMI"] >= 18.5 and dataframe["BMI"] <= 24.9:
        return "healthy_weight"
    elif dataframe["BMI"] >= 25 and dataframe["BMI"] <= 29.9:
        return "over_weight"
    elif dataframe["BMI"] >= 30:
        return "obese"

df = df.assign(BMI_Range=df.apply(calculate_bmi_range, axis=1))

df["pressure_ins"] = df["Insulin"] * df["BloodPressure"]

df["insulin_glucose"] = df["Insulin"] * df["Glucose"]

df["glucose_fun"] = df["Glucose"] / df["DiabetesPedigreeFunction"]

df["risk_tension"] = df["BloodPressure"] * df["Glucose"] / df["BMI"]
df["risk_tension_seg"] = pd.qcut(df["risk_tension"], 4, labels=["low_risk", "normal", "moderate_risk", "high_risk"])

df.loc[(df["Glucose"] < 140), 'glucose_cat'] = "normal"
df.loc[(df["Glucose"] >= 140), 'glucose_cat'] = "at_risk"

df.loc[(df["Insulin"] < 126) & (df["Age"] <= 50), "age_insulin_check"] = "normal_mature"
df.loc[(df["Insulin"] < 126) & (df["Age"] > 50), "age_insulin_check"] = "normal_elder"
df.loc[(df["Insulin"] >= 126) & (df["Age"] <= 50), "age_insulin_check"] = "not_normal_mature"
df.loc[(df["Insulin"] >= 126) & (df["Age"] > 50), "age_insulin_check"] = "not_normal_elder"

## Encoding ##
# After extracting all categorical variables, it was determined that they are ordinal in nature.
# Therefore, label encoding was chosen for the encoding process as it outperformed One-Hot encoding based on accuracy results.

cat_cols, num_cols, cat_but_car = grab_col_names(df)

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

# One-Hot Encoding

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, ohe_cols)

# Label Encoding

le = LabelEncoder()

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

ordinal_cols =
df = label_encoder(df, "glucose_cat")


for col in ohe_cols:
    df = label_encoder(df, col)


## Feature Scaling ##

robust_scaler = RobustScaler()
df[num_cols] = robust_scaler.fit_transform(df[num_cols])
df.head()


## Build Random Forest Classifier Model ##

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.30, random_state=42)

rf_model = RandomForestClassifier().fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_test, y_pred)

plot_importance(rf_model, X_train)


# Model Tuning

rf = RandomForestClassifier(max_depth=10, max_features=5, max_leaf_nodes=11, min_samples_split=3, n_estimators=1000)
rf_tuned = rf.fit(X_train, y_train)
y_pred = rf_tuned.predict(X_test)
accuracy_score(y_test, y_pred)

plot_importance(rf_model, X_train)

# The accuracy score is 0.88, and the most important feature is Insulin, while two of the top three features are extracted ones.
