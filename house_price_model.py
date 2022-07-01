from numpy import zeros
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot  as plt
import matplotlib.ticker as ticker
pd.set_option('max_columns', None)
pd.set_option('max_rows', None)

# constants
AVERAGE_SALE_TO_LIST_RATIO = 1.18
FEATURE_LIST = [
    # 'sq_ft',
    'bed',
    # 'bath',
    # 'walk_time_to_phinney_ridge',
    # 'walk_time_to_greenwood',
    'sq_ft_pow',
    'walk_time_to_phinney_ridge_pow',
    'walk_time_to_greenwood_pow',
    # 'combined_walk_time_2',
    ]

# helper functions
def _format_ticks(val):
    '''formats plot ticks'''
    if val >= 1000000:
        return('{:,.1f}M'.format(val/1000000))
    return('{:,.0f}k'.format(val/1000))

def _extract_features(df):
    '''extract features'''
    df['sq_ft_pow'] = df['sq_ft']**1
    df['walk_time_to_phinney_ridge_pow'] = df['walk_time_to_phinney_ridge']**2
    df['walk_time_to_greenwood_pow'] = df['walk_time_to_greenwood']**2
    # df['combined_walk_time_2'] = (df['walk_time_to_phinney_ridge']+df['walk_time_to_greenwood'])**2
    n_samples = len(df)
    n_features = len(FEATURE_LIST)
    X = np.zeros((n_samples,n_features))
    for i in range(n_features):
        X[:,i] = df[FEATURE_LIST[i]].to_list()
    # print(X)
    return(X)

def _extract_sale_price_target(sale_price,list_price):
    '''Extracts the sale price target'''
    if pd.isnull(sale_price):
        return(list_price*AVERAGE_SALE_TO_LIST_RATIO)
    return(sale_price)

def _extract_target_label(df):
    df['sale_price_target'] = df.apply(lambda row: _extract_sale_price_target(row['sale_price'],row['list_price']), axis=1)
    # print(df['sale_price_target'])
    y = np.zeros((len(df),1))
    y[:,0] = df['sale_price_target'].to_list()
    # print(y)
    return(y)


# import data
df = pd.read_csv (r'data.csv')
df = df.set_index('address')
# print (df.loc['9217 4TH AVE NW','sale_price'])
# print(df.index.to_list())

# extract features
df_train = df.copy()
df_train = df_train[df_train['status'] == 'Sold']
X_train = _extract_features(df_train)

# extract target labels
y_train = _extract_target_label(df_train)

# train
scaler = StandardScaler()
clf = LinearRegression().fit(scaler.fit_transform(X_train), y_train)
print('intercept: {:.2f}'.format(clf.intercept_[0]))
print('Coefficients')
for i in range(len(FEATURE_LIST)):
    print('\t{}: {:.2f}'.format(FEATURE_LIST[i],clf.coef_[0,i]))

# predict
X = _extract_features(df)
y = _extract_target_label(df)
y_pred = clf.predict(scaler.fit_transform(X))
error = np.array(y_pred - y)
# print(error)
avg_error = np.average(np.abs(error))
print("average abs error: {:.2f}".format(avg_error))

# plot
fig, ax = plt.subplots()
plt.scatter(y, y_pred)
plt.xlabel('Actual Sale Price [$]')
plt.ylabel('Predicted Sale Price [$]')
plt.rcParams["figure.figsize"] = (10,6) # Custom figure size in inches
plt.title("Seattle House Sale Price Predictions")
plt.xlim([800000, 1200000])
plt.ylim([800000, 1200000])
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: _format_ticks(val)))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: _format_ticks(val)))
plt.grid()
plt.show()

