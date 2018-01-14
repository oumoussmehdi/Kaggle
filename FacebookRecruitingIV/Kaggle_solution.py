import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = "C:\\Users\\OUMOUSS\\PycharmProjects\\Adomik\\"
bids = pd.read_csv(path + 'bids.csv')
bids.sort_values('time', axis=0, ascending=True, inplace=True)

train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
nrow_train = train.shape[0]
bidder_id = test['bidder_id']
y = train["outcome"]

print("start!")

def prepare_data(bids, train,test):

    # Features: distinct counts of different features in the bids dataset
    grouping_counts = bids.groupby('bidder_id').agg({
        'bid_id': 'nunique',
        'auction': 'nunique',
        'device': 'nunique',
        'country': 'nunique',
        'ip': 'nunique',
        'url': 'nunique'}).reset_index()

    # Feature: based on time column
    grouping_time = bids.groupby('bidder_id').agg({
        'time': ['min', 'max']
    }).reset_index()
    grouping_time.columns = ["_".join(x) for x in grouping_time.columns.ravel()]
    grouping_time = grouping_time.rename(columns={'bidder_id_': 'bidder_id'})
    grouping_time['time_diff'] = (grouping_time['time_max'] - grouping_time['time_min']) // 10 ** 9

    # Feature: common merchandise per bidder_id
    merchandise_count = bids[['bidder_id', 'merchandise']].groupby(
        ['bidder_id', 'merchandise']).merchandise.count().reset_index(name='count')
    common_merchandise_grouping = merchandise_count[['bidder_id', 'merchandise']].loc[
        merchandise_count.groupby(['bidder_id'], sort=False)['count'].idxmax()]
    common_merchandise_grouping = common_merchandise_grouping.rename(index=str,
                                                                  columns={"merchandise": "common_merchandise"})
    # Feature: common country per bidder_id
    country_count = bids[['bidder_id', 'country']].groupby(['bidder_id', 'country']).country.count().reset_index(
        name='count')
    common_country_grouping = country_count[['bidder_id', 'country']].loc[
        country_count.groupby(['bidder_id'], sort=False)['count'].idxmax()]
    common_country_grouping = common_country_grouping.rename(index=str, columns={"country": "common_country"})

    ## Train/Test dataset
    train.drop('outcome', axis=1, inplace=True)
    # concat train and test to apply the transformations in one go
    df = pd.concat([train, test], 0)
    df.drop('payment_account', axis=1, inplace=True)
    df.drop('address', axis=1, inplace=True)

    df = pd.merge(df, grouping_counts, on='bidder_id', how='left')
    df = pd.merge(df, grouping_time[['bidder_id', 'time_diff']], on='bidder_id', how='left')
    df = df.fillna(0)
    del grouping_counts, grouping_time

    df = pd.merge(df, common_merchandise_grouping, on='bidder_id', how='left')
    df = pd.merge(df, common_country_grouping, on='bidder_id', how='left')
    df = df.fillna('unknown')
    del common_merchandise_grouping, common_country_grouping

    df.drop('bidder_id', axis=1, inplace=True)
    X = df
    del df
    return X

X = prepare_data(bids,train,test)
del train, test
print(X)

# Encoding common_merchandise & common_country
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
X["common_merchandise"] = le.fit_transform(X["common_merchandise"])
X["common_country"] = le.fit_transform(X["common_country"])

### Preparing data
train = X[:nrow_train]
test = X[nrow_train:]
train = pd.concat([train, y], 1)

### Up-sample Minority Class: the data is imbalanced
# therefore we tought using a resampling technique is necessary
from sklearn.utils import resample

# Separate majority and minority classes
df_majority = train[train.outcome == 0]
df_minority = train[train.outcome == 1]

# Upsample minority class
df_minority_upsampled = resample(df_minority,
                                 replace=True,  # sample with replacement
                                 n_samples=1910,  # to match majority class
                                 random_state=123)  # reproducible results

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

train = df_upsampled

print(" model!")

### Model
from sklearn.ensemble import RandomForestClassifier

rf_params = {'n_estimators': 20,
             'max_depth': None,
             'min_samples_split': 2,
             'random_state': 0}

model = RandomForestClassifier(**rf_params)

from sklearn.model_selection import train_test_split, cross_val_score

cols = ['country', 'ip', 'bid_id', 'device', 'url', 'auction',
        'time_diff', 'common_merchandise', 'common_country']

X_train, X_test, y_train, y_test = train_test_split(train[cols],
                                                    train.outcome,
                                                    test_size=0.3,
                                                    # stratify=train.outcome,
                                                    random_state=0)

model.fit(X_train, y_train)

# Having a look to the features importances
features = X_train.columns
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

indices = np.argsort(importances)

plt.figure(1)
plt.title('Feature Importances')
plt.barh(range(len(indices)),
         importances[indices],
         color='b',
         align='center')
plt.yticks(range(len(indices)), features[indices])
plt.xlabel('Relative Importance')

### Evaluation
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve, auc

model.fit(X_train, y_train)
y_score = model.predict(X_test)
print("roc_auc_score: %.3f" % roc_auc_score(y_test, y_score))
fpr, tpr, thresholds = roc_curve(y_test, y_score, pos_label=1)
print(fpr, tpr, thresholds)

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, y_score))

cm = confusion_matrix(y_test, y_score, labels=[0, 1])
print(cm)
print("we have ", cm[0][0] + cm[1][1], " correct predictions and ", cm[1][0] + cm[0][1], " incorrect predictions.")


def plot_roc(fpr, tpr, roc_auc):
    plt.figure()
    plt.plot(fpr,
             tpr,
             color='darkorange',
             label='ROC curve (area = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()


y_score_proba = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_score_proba)
print("roc_auc_score: %.3f" % roc_auc)
fpr, tpr, thresholds = roc_curve(y_test, y_score_proba)

plot_roc(fpr, tpr, roc_auc)

## Submission

# proba_predictioncs = model.predict_proba(test)[:,1]
prediction = model.predict(test)
prediction.shape


def submit(bidder_id, prediction, filename):
    submission = pd.DataFrame(np.column_stack([bidder_id, prediction]), columns=["bidder_id", "prediction"])
    submission["bidder_id"] = submission["bidder_id"]
    submission["prediction"] = submission["prediction"]
    submission.to_csv(filename, index=False)


submit(bidder_id, prediction, 'submission.csv')
print("Done!")