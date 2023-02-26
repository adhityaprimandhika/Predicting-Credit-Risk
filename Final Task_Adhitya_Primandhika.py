# For visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# For data engineering
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import MinMaxScaler

# For machine learning and its evaluation
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RepeatedStratifiedKFold
from imblearn.over_sampling import SMOTE
from pickle import dump, load
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as imbpipeline

# For ignoring some warnings
import warnings
warnings.filterwarnings("ignore")

# Looking to our dataset
df_loan = pd.read_csv("loan_data_2007_2014.csv", index_col=0)
df_loan

# Find the information about dataset
df_loan.info()

# Dropping some columns with only NaN values
df_loan = df_loan.drop(["annual_inc_joint", "dti_joint", "verification_status_joint", \
                        "open_acc_6m", "open_il_6m", "open_il_12m", "open_il_24m", \
                        "mths_since_rcnt_il", "total_bal_il", "il_util", "open_rv_12m", \
                        "open_rv_24m", "max_bal_bc", "all_util", "inq_fi", "total_cu_tl", \
                        "inq_last_12m"], axis=1)

df_loan.head()

# Looking for total of unique values from each column
df_loan.nunique()

# Dropping some columns that not so useful and redundant
df_loan = df_loan.drop(["id","member_id","url","zip_code","desc","addr_state","policy_code", \
                        "application_type", "title", "emp_title", "sub_grade"], axis=1)

df_loan.head()

# Dropping some columns that have data from the future that we don"t know yet
df_loan = df_loan.drop(["out_prncp", "out_prncp_inv", "recoveries", "last_pymnt_d", \
                        "next_pymnt_d", "last_pymnt_amnt", "total_pymnt", "total_pymnt_inv", \
                        "total_rec_prncp", "total_rec_int", "total_rec_late_fee", \
                        "collection_recovery_fee", "funded_amnt", "funded_amnt_inv"], axis=1)

df_loan.head()

df_loan.columns

# Percentage of null values for each column
df_loan.isnull().mean()

# Dropping columns with null values > 30%
df_loan = df_loan.drop(["mths_since_last_delinq", "mths_since_last_major_derog", \
                        "mths_since_last_record"], axis=1)

df_loan.head()

# Total of null values for each column
df_loan.isnull().sum()

# See the unique values
print(df_loan["emp_length"].unique())
print(df_loan["earliest_cr_line"].unique())
print(df_loan["last_credit_pull_d"].unique())

# Fill null values with mode of those columns
df_loan["emp_length"].fillna(df_loan["emp_length"].mode()[0], inplace=True)
df_loan["earliest_cr_line"].fillna(df_loan["earliest_cr_line"].mode()[0], inplace=True)
df_loan["last_credit_pull_d"].fillna(df_loan["last_credit_pull_d"].mode()[0], inplace=True)

# Check if we already handle it
df_loan.isnull().sum()

# Plot the distribution of annual_inc
sns.displot(df_loan, x="annual_inc", kind="kde")
plt.title("Distribution of annual_inc")
plt.show()

# Plot the distribution of delinq_2yrs
sns.displot(df_loan, x="delinq_2yrs", kind="kde")
plt.title("Distribution of delinq_2yrs")
plt.show()

# Plot the distribution of inq_last_6mths
sns.displot(df_loan, x="inq_last_6mths", kind="kde")
plt.title("Distribution of inq_last_6mths")
plt.show()

# Plot the distribution of open_acc
sns.displot(df_loan, x="open_acc", kind="kde")
plt.title("Distribution of open_acc")
plt.show()

# Plot the distribution of pub_rec
sns.displot(df_loan, x="pub_rec", kind="kde")
plt.title("Distribution of pub_rec")
plt.show()

# Plot the distribution of revol_util
sns.displot(df_loan, x="revol_util", kind="kde")
plt.title("Distribution of revol_util")
plt.show()

# Plot the distribution of total_acc
sns.displot(df_loan, x="total_acc", kind="kde")
plt.title("Distribution of total_acc")
plt.show()

# Plot the distribution of collections_12_mths_ex_med
sns.displot(df_loan, x="collections_12_mths_ex_med", kind="kde")
plt.title("Distribution of collections_12_mths_ex_med")
plt.show()

# Plot the distribution of acc_now_delinq
sns.displot(df_loan, x="acc_now_delinq", kind="kde")
plt.title("Distribution of acc_now_delinq")
plt.show()

# Plot the distribution of tot_coll_amt
sns.displot(df_loan, x="tot_coll_amt", kind="kde")
plt.title("Distribution of tot_coll_amt")
plt.show()

# Plot the distribution of tot_cur_bal
sns.displot(df_loan, x="tot_cur_bal", kind="kde")
plt.title("Distribution of tot_cur_bal")
plt.show()

# Plot the distribution of total_rev_hi_lim
sns.displot(df_loan, x="total_rev_hi_lim", kind="kde")
plt.title("Distribution of total_rev_hi_lim")
plt.show()

# Fill null values with median of those columns
df_loan["annual_inc"].fillna(df_loan["annual_inc"].median(), inplace=True)
df_loan["delinq_2yrs"].fillna(df_loan["delinq_2yrs"].median(), inplace=True)
df_loan["inq_last_6mths"].fillna(df_loan["inq_last_6mths"].median(), inplace=True)
df_loan["open_acc"].fillna(df_loan["open_acc"].median(), inplace=True)
df_loan["pub_rec"].fillna(df_loan["pub_rec"].median(), inplace=True)
df_loan["revol_util"].fillna(df_loan["revol_util"].median(), inplace=True)
df_loan["total_acc"].fillna(df_loan["total_acc"].median(), inplace=True)
df_loan["collections_12_mths_ex_med"].fillna(df_loan["collections_12_mths_ex_med"].median(), inplace=True)
df_loan["acc_now_delinq"].fillna(df_loan["acc_now_delinq"].median(), inplace=True)
df_loan["tot_coll_amt"].fillna(df_loan["tot_coll_amt"].median(), inplace=True)
df_loan["tot_cur_bal"].fillna(df_loan["tot_cur_bal"].median(), inplace=True)
df_loan["total_rev_hi_lim"].fillna(df_loan["total_rev_hi_lim"].median(), inplace=True)

# Check if we already handle it
df_loan.isnull().sum()

# Check the unique values in loan status column
df_loan["loan_status"].unique()

# Filter data with the value of loan_status is equal to "Fully Paid" or "Charged Off"
df_loan = df_loan[(df_loan["loan_status"] == "Fully Paid") | (df_loan["loan_status"] == "Charged Off")]

# Create a dictionary for mapping the loan_status value
loan_status_dict = {"Fully Paid":"Good Loan", "Charged Off":"Bad Loan"}

# Mapping loan_status column values
df_loan["loan_status"] = df_loan["loan_status"].map(loan_status_dict).astype(str)

df_loan.head()

# Create a dictionary for mapping the bad_loan value
bad_loan_dict = {"Good Loan": 0, "Bad Loan": 1}

# Create new column bad_loan
df_loan["bad_loan"] = df_loan["loan_status"].copy()

# Mapping bad_loan column values
df_loan["bad_loan"] = df_loan["bad_loan"].map(bad_loan_dict).astype(int)

# Drop loan_status column
df_loan = df_loan.drop(["loan_status"], axis=1)

df_loan.head()

# Create correlation matrix
df_loan.corr().style.background_gradient(cmap="viridis")

# Create correlation matrix
plt.figure(figsize=(20,10))
sns.heatmap(df_loan.corr(), annot=True, cmap="viridis")
plt.title("Correlation Matrix")
plt.show()

# Check data type every column
df_loan.info()

# See the unique values
print(df_loan["term"].unique())
print(df_loan["grade"].unique())
print(df_loan["emp_length"].unique())
print(df_loan["home_ownership"].unique())
print(df_loan["verification_status"].unique())
print(df_loan["issue_d"].unique())
print(df_loan["pymnt_plan"].unique())
print(df_loan["purpose"].unique())
print(df_loan["earliest_cr_line"].unique())
print(df_loan["initial_list_status"].unique())
print(df_loan["last_credit_pull_d"].unique())

# Transform term column
df_loan["term"] = df_loan["term"].str.replace(" 36 months", "36")
df_loan["term"] = df_loan["term"].str.replace(" 60 months", "60")
df_loan["term"] = df_loan["term"].astype(int)

# Transform grade column
# Creating instance of labelencoder
le_grade = LabelEncoder()
df_loan["grade"] = le_grade.fit_transform(df_loan["grade"])

# Transform emp_length column
df_loan["emp_length"] = df_loan["emp_length"].str.replace("< 1 year", str(0))
df_loan["emp_length"] = df_loan["emp_length"].str.replace("\+ years", "")
df_loan["emp_length"] = df_loan["emp_length"].str.replace(" years", "")
df_loan["emp_length"] = df_loan["emp_length"].str.replace(" year", "")
df_loan["emp_length"] = df_loan["emp_length"].astype(int)

# Transform home_ownership column
# Creating instance of labelencoder
le_home_ownership = LabelEncoder()
df_loan["home_ownership"] = le_home_ownership.fit_transform(df_loan["home_ownership"])

# Transform verification_status column
# Creating instance of labelencoder
le_verification_status = LabelEncoder()
df_loan["verification_status"] = le_verification_status.fit_transform(df_loan["verification_status"])

# Transform issue_d column
df_loan["issue_d"] = df_loan["issue_d"].str.replace("Jan-", "")
df_loan["issue_d"] = df_loan["issue_d"].str.replace("Feb-", "")
df_loan["issue_d"] = df_loan["issue_d"].str.replace("Mar-", "")
df_loan["issue_d"] = df_loan["issue_d"].str.replace("Apr-", "")
df_loan["issue_d"] = df_loan["issue_d"].str.replace("May-", "")
df_loan["issue_d"] = df_loan["issue_d"].str.replace("Jun-", "")
df_loan["issue_d"] = df_loan["issue_d"].str.replace("Jul-", "")
df_loan["issue_d"] = df_loan["issue_d"].str.replace("Aug-", "")
df_loan["issue_d"] = df_loan["issue_d"].str.replace("Sep-", "")
df_loan["issue_d"] = df_loan["issue_d"].str.replace("Oct-", "")
df_loan["issue_d"] = df_loan["issue_d"].str.replace("Nov-", "")
df_loan["issue_d"] = df_loan["issue_d"].str.replace("Dec-", "")
df_loan["issue_d"] = df_loan["issue_d"].astype(int)

# Transform pymnt_plan column
# Creating instance of labelencoder
le_pymnt_plan = LabelEncoder()
df_loan["pymnt_plan"] = le_pymnt_plan.fit_transform(df_loan["pymnt_plan"])

# Transform purpose column
# Creating instance of labelencoder
le_purpose = LabelEncoder()
df_loan["purpose"] = le_purpose.fit_transform(df_loan["purpose"])

# Transform earliest_cr_line column
df_loan["earliest_cr_line"] = df_loan["earliest_cr_line"].str.replace("Jan-", "")
df_loan["earliest_cr_line"] = df_loan["earliest_cr_line"].str.replace("Feb-", "")
df_loan["earliest_cr_line"] = df_loan["earliest_cr_line"].str.replace("Mar-", "")
df_loan["earliest_cr_line"] = df_loan["earliest_cr_line"].str.replace("Apr-", "")
df_loan["earliest_cr_line"] = df_loan["earliest_cr_line"].str.replace("May-", "")
df_loan["earliest_cr_line"] = df_loan["earliest_cr_line"].str.replace("Jun-", "")
df_loan["earliest_cr_line"] = df_loan["earliest_cr_line"].str.replace("Jul-", "")
df_loan["earliest_cr_line"] = df_loan["earliest_cr_line"].str.replace("Aug-", "")
df_loan["earliest_cr_line"] = df_loan["earliest_cr_line"].str.replace("Sep-", "")
df_loan["earliest_cr_line"] = df_loan["earliest_cr_line"].str.replace("Oct-", "")
df_loan["earliest_cr_line"] = df_loan["earliest_cr_line"].str.replace("Nov-", "")
df_loan["earliest_cr_line"] = df_loan["earliest_cr_line"].str.replace("Dec-", "")
df_loan["earliest_cr_line"] = df_loan["earliest_cr_line"].astype(int)

# Transform initial_list_status column
# Creating instance of labelencoder
le_initial_list_status = LabelEncoder()
df_loan["initial_list_status"] = le_initial_list_status.fit_transform(df_loan["initial_list_status"])

# Transform last_credit_pull_d column
df_loan["last_credit_pull_d"] = df_loan["last_credit_pull_d"].str.replace("Jan-", "")
df_loan["last_credit_pull_d"] = df_loan["last_credit_pull_d"].str.replace("Feb-", "")
df_loan["last_credit_pull_d"] = df_loan["last_credit_pull_d"].str.replace("Mar-", "")
df_loan["last_credit_pull_d"] = df_loan["last_credit_pull_d"].str.replace("Apr-", "")
df_loan["last_credit_pull_d"] = df_loan["last_credit_pull_d"].str.replace("May-", "")
df_loan["last_credit_pull_d"] = df_loan["last_credit_pull_d"].str.replace("Jun-", "")
df_loan["last_credit_pull_d"] = df_loan["last_credit_pull_d"].str.replace("Jul-", "")
df_loan["last_credit_pull_d"] = df_loan["last_credit_pull_d"].str.replace("Aug-", "")
df_loan["last_credit_pull_d"] = df_loan["last_credit_pull_d"].str.replace("Sep-", "")
df_loan["last_credit_pull_d"] = df_loan["last_credit_pull_d"].str.replace("Oct-", "")
df_loan["last_credit_pull_d"] = df_loan["last_credit_pull_d"].str.replace("Nov-", "")
df_loan["last_credit_pull_d"] = df_loan["last_credit_pull_d"].str.replace("Dec-", "")
df_loan["last_credit_pull_d"] = df_loan["last_credit_pull_d"].astype(int)

# Check our data
df_loan.head()

# Prepare X and y
X = df_loan.drop(["bad_loan"], axis=1)
y = df_loan["bad_loan"]

# See the shape of it
X.shape, y.shape

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print("Data split done")

X_train.shape, X_test.shape

# List of models
models = []
models.append(("Extra Trees", ExtraTreesClassifier()))
models.append(("Random Forest", RandomForestClassifier()))
models.append(("XGBoost", XGBClassifier()))
models.append(("Decision Tree", DecisionTreeClassifier()))
models.append(("Logistic Regression", LogisticRegression()))

results = []
names = []
scoring = "accuracy"
for name, model in models:
    pipeline = imbpipeline(steps = [["smote", SMOTE(random_state=11)], ["scaler", MinMaxScaler()], ["classifier", model]])
    kfold = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=42)
    cv_results = cross_val_score(pipeline, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Show the comparisons through visualization
fig = plt.figure(figsize=(12, 6))
fig.suptitle("Algorithm Comparison")
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.grid()
plt.show()

# PCA initialization
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA().fit(X_scaled)

# Create plot for explained variance ratio
plt.rcParams["figure.figsize"] = (12,6)
fig, ax = plt.subplots()
xi = np.arange(1, 29, step=1)
yi = np.cumsum(pca.explained_variance_ratio_)

plt.ylim(0.0,1.1)
plt.plot(xi, yi, marker="o", linestyle="--", color="b")

plt.xlabel("Number of Components")
plt.xticks(np.arange(0, 29, step=1)) #change from 0-based array index to 1-based human-readable label
plt.ylabel("Cumulative variance (%)")
plt.title("The number of components needed to explain variance")

plt.axhline(y=0.95, color="r", linestyle="-")
plt.text(0.5, 0.96, "95% cut-off threshold", color = "red")

ax.grid(axis="x")
plt.show()

# Create model pipeline
model = imbpipeline(steps = [["smote", SMOTE(random_state=11)], ["scaler", MinMaxScaler()], \
                                ["pca", PCA(n_components=10)], ["classifier", XGBClassifier()]])

# Cross validation
kfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=42)
cv_results = cross_val_score(pipeline, X, y, cv=kfold, scoring="accuracy")

# Cross validation score
print("Score : {}".format(round(cv_results.mean(),2)))

# Save model
filename = "finalized_model.sav"
dump(model, open(filename, "wb"))