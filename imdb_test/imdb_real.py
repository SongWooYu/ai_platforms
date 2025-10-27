import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os

NUM_WORDS = 1000
(train_data, train_labels), (test_data, test_labels) = keras.datasets.imdb.load_data(num_words=NUM_WORDS)

def multi_hot_sequences(sequences, dimension):
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0
    return results

train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)

# plt.plot(train_data[0])
# plt.show()
train_labels = np.reshape(train_labels, (-1, 1))
test_labels = np.reshape(test_labels, (-1, 1))

baseline_model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
baseline_model.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy', 'binary_crossentropy'])
baseline_model.summary()

baseline_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join('./imdb_checkpoints', 'baseline_model.keras'),
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)






smaller_model = keras.Sequential([
    keras.layers.Dense(4, activation='relu', input_shape=(NUM_WORDS,)),
    keras.layers.Dense(4, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

smaller_model.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy', 'binary_crossentropy'])
smaller_model.summary()

smaller_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join('./imdb_checkpoints', 'smaller_model.keras'),
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)






bigger_model = keras.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(NUM_WORDS,)),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

bigger_model.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy', 'binary_crossentropy'])
bigger_model.summary()

bigger_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join('./imdb_checkpoints', 'bigger_model.keras'),
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)





l2_model = keras.Sequential([
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                       activation='relu', input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                       activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

l2_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'binary_crossentropy'])
l2_model.summary()

l2_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join('./imdb_checkpoints', 'l2_model.keras'),
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)






dpt_model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(NUM_WORDS,)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])
dpt_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'binary_crossentropy'])
dpt_model.summary()
dpt_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join('./imdb_checkpoints', 'dpt_model.keras'),
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)








l2dpt_model = keras.Sequential([
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                       activation='relu', input_shape=(NUM_WORDS,)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                       activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation='sigmoid')
])
l2dpt_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'binary_crossentropy'])
l2dpt_model.summary()
l2dpt_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join('./imdb_checkpoints', 'l2dpt_model.keras'),
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)










baseline_history = baseline_model.fit(train_data,
                                      train_labels,
                                      epochs=200,
                                      batch_size=512,
                                      validation_data=(test_data, test_labels),
                                      callbacks=[baseline_checkpoint],
                                      verbose=2)

smaller_history = smaller_model.fit(train_data,
                                    train_labels,
                                    epochs=200,
                                    batch_size=512,
                                    validation_data=(test_data, test_labels),
                                    callbacks=[smaller_checkpoint],
                                    verbose=2)

bigger_history = bigger_model.fit(train_data,
                                  train_labels,
                                  epochs=200,
                                  batch_size=512,
                                  validation_data=(test_data, test_labels),
                                  callbacks=[bigger_checkpoint],
                                  verbose=2)

l2_history = l2_model.fit(train_data, train_labels,
                                epochs=200,
                                batch_size=512,
                                validation_data=(test_data, test_labels),
                                callbacks=[l2_checkpoint],
                                verbose=2)

dpt_history = dpt_model.fit(train_data, train_labels,
                                epochs=200,
                                batch_size=512,
                                validation_data=(test_data, test_labels),
                                callbacks=[dpt_checkpoint],
                                verbose=2)

l2dpt_history = l2dpt_model.fit(train_data, train_labels,
                                epochs=200,
                                batch_size=512,
                                validation_data=(test_data, test_labels),
                                callbacks=[l2dpt_checkpoint],
                                verbose=2)




def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(16,10))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],
                       '--', label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                 label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()

    plt.xlim([0,max(history.epoch)])
    plt.show()




plot_history([('baseline', baseline_history),
              ('smaller', smaller_history),
              ('bigger', bigger_history),
              ('l2', l2_history),
              ('dropout', dpt_history),
              ('l2_dropout', l2dpt_history)])

plot_history([('baseline', baseline_history),
              ('smaller', smaller_history),
              ('bigger', bigger_history),
              ('l2', l2_history),
              ('dropout', dpt_history),
              ('l2_dropout', l2dpt_history)], key='accuracy')


print("\n--- Loading and evaluating best models ---")

model_dir = './imdb_checkpoints'

best_baseline_model = tf.keras.models.load_model('best_baseline_model.keras')
loss, acc = best_baseline_model.evaluate(test_data, test_labels, verbose=0)
print(f"Best Baseline Model Accuracy: {100*acc:5.2f}%")

best_smaller_model = tf.keras.models.load_model('best_smaller_model.keras')
loss, acc = best_smaller_model.evaluate(test_data, test_labels, verbose=0)
print(f"Best Smaller Model Accuracy: {100*acc:5.2f}%")

best_bigger_model = tf.keras.models.load_model('best_bigger_model.keras')
loss, acc = best_bigger_model.evaluate(test_data, test_labels, verbose=0)
print(f"Best Bigger Model Accuracy: {100*acc:5.2f}%")

best_l2_model = tf.keras.models.load_model('best_l2_model.keras')
loss, acc = best_l2_model.evaluate(test_data, test_labels, verbose=0)
print(f"Best L2 Model Accuracy: {100*acc:5.2f}%")

best_dpt_model = tf.keras.models.load_model('best_dpt_model.keras')
loss, acc = best_dpt_model.evaluate(test_data, test_labels, verbose=0)
print(f"Best Dropout Model Accuracy: {100*acc:5.2f}%")

best_l2dpt_model = tf.keras.models.load_model('best_l2dpt_model.keras')
loss, acc = best_l2dpt_model.evaluate(test_data, test_labels, verbose=0)
print(f"Best L2+Dropout Model Accuracy: {100*acc:5.2f}%")
    

