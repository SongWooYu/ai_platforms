from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

vocabulary_size = 10000
num_tags = 100
num_departments = 4

title = keras.Input(shape=(vocabulary_size,), name='title')
text_body = keras.Input(shape=(vocabulary_size,), name='text_body')
tags = keras.Input(shape=(num_tags,), name='tags')

# 각 입력에 대해 별도의 처리 층을 만듦
title_feature = layers.Dense(128, activation='relu')(title)
text_body_feature = layers.Dense(128, activation='relu')(text_body)
tags_feature = tags
# 처리 층을 거친 입력을 하나로 합침
features = layers.Concatenate()([title_feature, text_body_feature, tags_feature])
features = layers.Dense(1024, activation='relu')(features)

# 우선순위 예측을 위한 출력 층
priority = layers.Dense(128, activation="sigmoid", name='priority')(features)
priority = layers.Dense(1, activation='sigmoid')(priority)

# 부서 예측을 위한 출력 층
department_1 = layers.Dense(128, activation='relu')(features)
department_2 = layers.Dense(128, activation='relu')(department_1)
department = layers.Dense(num_departments, activation="softmax", name='department')(department_2)

# 모델 정의
model = keras.Model(inputs=[title, text_body, tags],
                    outputs=[priority, department])



num_samples = 1280

title_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
text_body_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))

priority_data = np.random.randint(0, 2, size=(num_samples, 1))
department_data = np.random.randint(0, 2, size=(num_samples, num_departments))


model.compile(optimizer='rmsprop',
              loss=["mean_squared_error", "categorical_crossentropy"],
              # 여러 출력에 대해 서로 다른 메트릭을 지정할 수 있기에 리스트에 리스트를 넣어줌
              metrics=[["mean_absolute_error"], ["accuracy"]])
# 순서를 맞춰줘야 함. 그렇지 않다면 딕셔너리를 사용
model.fit([title_data, text_body_data, tags_data],
          [priority_data, department_data],
          epochs=1,)
print(model.evaluate([title_data, text_body_data, tags_data],
               [priority_data, department_data]))
priority_preds, department_preds = model.predict([title_data, text_body_data, tags_data])
print(priority_preds, department_preds)

keras.utils.plot_model(model, "ticket_classifier.png")