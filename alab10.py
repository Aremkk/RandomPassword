import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import random

# 1. Генерация данных
def generate_passwords(num_passwords, password_length):
    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
    passwords = []
    for _ in range(num_passwords):
      password = ''.join(random.choice(chars) for _ in range(password_length))
      passwords.append(password)

    # Добавляем часто встречающиеся пароли
    common_passwords = ["password123", "12345678", "qwert", "Abcde123!", "Date12345"]
    passwords += common_passwords

    return passwords

# 2. Преобразование данных
def prepare_data(passwords, chars):
    char_to_int = dict((c, i) for i, c in enumerate(chars))
    int_to_char = dict((i, c) for i, c in enumerate(chars))
    num_chars = len(chars)

    encoded_passwords = []
    encoded_target = []
    for password in passwords:
      for i in range(len(password) - 1):
          encoded_pass = [char_to_int[char] for char in password[:i+1]]
          encoded_passwords.append(encoded_pass)
          encoded_target.append(char_to_int[password[i+1]])


    max_len = max([len(x) for x in encoded_passwords])

    padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(encoded_passwords, maxlen=max_len, padding='pre')

    return padded_inputs, np.array(encoded_target), num_chars, int_to_char, max_len, char_to_int

# 3. Построение нейросети
def create_model(num_chars, max_len):
    model = Sequential()
    model.add(Embedding(num_chars, 32, input_length = max_len))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dense(num_chars, activation='softmax')) # Выходной слой с softmax
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# 4. Обучение нейросети
def train_model(model, X_train, y_train, X_val, y_val):
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])
    return model

# 5. Генерация паролей
def generate_password_with_lstm(model, chars, int_to_char, password_length, max_len, char_to_int, temperature = 1.0):
  generated_password = ""
  for _ in range(password_length):
    encoded_pass = [char_to_int[char] for char in generated_password]
    encoded_pass = tf.keras.preprocessing.sequence.pad_sequences([encoded_pass], maxlen=max_len, padding='pre')

    pred = model.predict(encoded_pass)
    pred = np.asarray(pred).astype('float64')

    # Apply temperature
    pred = np.log(pred) / temperature
    exp_preds = np.exp(pred)
    pred = exp_preds / np.sum(exp_preds)

    next_char_index = np.random.choice(len(chars), p=pred.flatten())
    generated_password += int_to_char[next_char_index]

  return generated_password

def check_password(generated_password, target_password):
   return generated_password == target_password

if __name__ == "__main__":
    password_length = 8
    num_passwords = 10000
    target_password = "password" # Целевой пароль

    chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"

    # 1. Генерация данных
    passwords = generate_passwords(num_passwords, password_length)

    # 2. Преобразование данных
    X, y, num_chars, int_to_char, max_len, char_to_int = prepare_data(passwords, chars)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Построение нейросети
    model = create_model(num_chars, max_len)
    # 4. Обучение нейросети
    model = train_model(model, X_train, y_train, X_val, y_val)

    # 5. Генерация и проверка паролей
    max_attempts = 100
    for attempt in range(max_attempts):
        generated_password = generate_password_with_lstm(model, chars, int_to_char, password_length, max_len, char_to_int, temperature=0.7)
        if check_password(generated_password, target_password):
            print(f"Пароль найден! '{generated_password}' за {attempt+1} попыток")
            break

        if attempt == max_attempts -1:
             print("Целевой пароль не был найден за 100 попыток")