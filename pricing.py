import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf


from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

np.set_printoptions(precision=3, suppress=True)

print(tf.__version__)

path = r'C:\Users\zajic\PycharmProjects\pricing\pricing.csv'

column_names = ['Fee', 'Complexity', 'Country', 'Deliverable']

dataset = pd.read_csv(path, names=column_names, sep=',')
# print(dataset)

# one-hot encoding
dataset['Country'] = dataset['Country'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, columns=['Country'], prefix='', prefix_sep='')

dataset['Deliverable'] = dataset['Deliverable'].map({1: 'FS', 2: 'GAAP', 3: 'Filing'})
dataset = pd.get_dummies(dataset, columns=['Deliverable'], prefix='', prefix_sep='')
#print(dataset)

# divide into test and train sets
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# show relations in train dataset
sns.pairplot(train_dataset[['Fee', 'Complexity']], diag_kind='kde')
plt.show()

# print(train_dataset.describe().transpose())
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('Fee')
test_labels = test_features.pop('Fee')

# print(train_dataset.describe().transpose()[['mean', 'std']])

# normalization
normalizer = preprocessing.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
# print(normalizer.mean.numpy())

# model
def build_and_compile_model(norm):
    model = tf.keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))
    return model


def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [Fee]')
  plt.legend()
  plt.grid(True)

# model

dnn_model = build_and_compile_model(normalizer)
print('-----')
dnn_model.summary()


history = dnn_model.fit(
    train_features, train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)

plot_loss(history)

plt.show()

test_results = {}
test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)

#pd.DataFrame(test_results, index=['Mean absolute error [Fee]']).T

test_predictions = dnn_model.predict(test_features).flatten()
plt.show()
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [Fee]')
plt.ylabel('Predictions [Fee]')

lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)

_ = plt.plot(lims, lims)
plt.show()
print(test_features)
print(test_predictions)
result = test_features
result['Fee'] = test_predictions
print(result)
result.to_csv('result.csv')