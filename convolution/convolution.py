from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# h, w, c = 28, 28, 1
inputs = keras.Input(shape=(28, 28, 1))
# filter의 차원을 32로 설정하고, kernel_size를 3으로 설정하여 3*3씩 슬라이딩, 26*26(패딩을 두느냐 안 두느냐에 따라 각 가로 세로에 -2씩 됨)의 32채널로 특성맵 출력
x = layers.Conv2D(filters=32, kernel_size=3, activation='relu')(inputs)
# 2*2로 스트라이드(하나씩 이동하므로)하는 맥스풀링으로 특성맵 크기를 절반으로 축소(16*16*32)
x = layers.MaxPooling2D(pool_size=2)(x)
# 다시 컨볼루션 층과 맥스풀링 층을 쌓음(13*13*64 -> 6*6*64)
x = layers.Conv2D(filters=64, kernel_size=3, activation='relu')(x)
x = layers.MaxPooling2D(pool_size=2)(x)
# 다시 컨볼루션 층을 쌓음(4*4*128)
x = layers.Conv2D(filters=128, kernel_size=3, activation='relu')(x)
x = layers.Flatten()(x)
# 완전 연결 층으로 연결
# x = layers.Dense(64, activation='relu')(x)

outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

# data 수집 및 전처리
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# 1채널로 차원 추가
train_images = train_images.reshape((60000, 28, 28, 1))
# int형을 float32형으로 변환 및 정규화
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype("float32") / 255
# 모델 컴파일
model.compile(optimizer='rmsprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 모델 훈련(64개를 한 묶음으로 6만개를 반복, 5회 반복)
model.fit(train_images, train_labels, epochs=5, batch_size=64)

# 모델 평가
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"테스트 정확도: {test_acc:.3f}")