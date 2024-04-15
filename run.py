import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding
from keras.optimizers import SGD, Adam
import matplotlib.pyplot as plt
import numpy as np
import csv 


def visualize(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def plot_real_vs_predicted(y_test_orig, y_pred):
    plt.scatter(y_test_orig, y_pred)
    plt.xlabel('Real Values')
    plt.ylabel('Predicted Values')
    plt.title('Real vs Predicted Values')
    plt.show()



data = pd.read_csv('iphi2802.csv', sep='\t') 

# Tokenization
texts = [' '.join(row.split()) for row in data['text']]
texts = [text.replace('[', '').replace('-', '').replace(']', '').replace('.', '') for text in texts]



# Apply TF-IDF encoding
tfidf_vectorizer = TfidfVectorizer(min_df=2, max_df=0.8)
X = tfidf_vectorizer.fit_transform(texts)





# Convert the matrix X to a DataFrame
X_df = pd.DataFrame(X.toarray())
non_zero_cols_indices = np.where(X_df.iloc[1] != 0)[0]


# scale which i ended up not using 
vects = X.toarray()
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(vects)
X_df = pd.DataFrame(X_scaled)

# Here I am removing the rows which ended up being zero after TF_IDF vectorization and its min_df/max_df parameters 
non_zero_cols_indices = np.where(X_df.iloc[4] != 0)[0]  



X_scaled = X # I did not end up using scaling 


# Calculate the sum of TF-IDF values for each row
row_sums = X_scaled.sum(axis=1)

# Find the indices of rows with non-zero sums
non_zero_indices = np.where(row_sums != 0)[0]

# Filter X_scaled to keep only the rows with non-zero sums
X_scaled = X_scaled[non_zero_indices]



kf = KFold(n_splits=5, shuffle=True)

# I am calculating the average of the date
date_min = data['date_min'].values
date_max = data['date_max'].values
y = (date_min + date_max) / 2 


# Scaling which I ended up not using 
scaler_Y = MinMaxScaler()
y_scaled = scaler_Y.fit_transform(y.reshape(-1, 1))
y_scaled = y.reshape(-1,1)



y_scaled = y # I did not end up using scaling 
y_scaled = y_scaled[non_zero_indices]


# Debug
print(f'here:{X[2]}')
print(' '.join(tfidf_vectorizer.inverse_transform(X[0])[0]))
print(tfidf_vectorizer.inverse_transform(X[-5]))
print(f'shape{X.shape}')
print(f'vshape{vects.shape}')
print(f'yshape{y_scaled.shape}')

learning_rate = 0.1
momentum = 0.6

optimizer = Adam(learning_rate=learning_rate, beta_1=momentum)

model = Sequential()


model.add(Dense(128, activation="relu"))
model.add(Dropout(1 - 0.8))  
model.add(Dense(64, activation="relu"))
model.add(Dropout(1 - 0.2))  
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer=optimizer)

total_rmse = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
    # Split data into train and test sets
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y_scaled[train_idx], y_scaled[test_idx]

    history = model.fit(X_train, y_train, epochs=200, batch_size=2000, validation_data=(X_test, y_test), verbose=0)
    print(f"Fold {fold + 1}: Train set size: {len(train_idx)}, Test set size: {len(test_idx)}")
    scores = model.evaluate(X_test, y_test, verbose=0)
    

    print("Fold :", fold, " RMSE:", scores)

    total_rmse.append(scores)

    if fold == 4:
        visualize(history)
    # Predictions
    y_pred = model.predict(X_test)
   
    if fold == 4:
        plot_real_vs_predicted(y_test, y_pred)
    # Print real and predicted values
    print("Real\tPredicted")
    for i in range(10):
        print(f"{y_test[i].reshape(1, -1)[0]}\t{y_pred[i][0]}")

average_rmse = np.mean(total_rmse)
print(f'mean_rmse:{average_rmse}')