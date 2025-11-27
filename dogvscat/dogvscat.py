import os, shutil, pathlib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt


original_dir = pathlib.Path("train")
new_base_dir = pathlib.Path("cats_vs_dogs_small")
def make_subset(subset_name, start_index, end_index):
    # 2번(dog, cat) 돌아감, 자료형은 str
    for category in ("cat", "dog"):
        # str을 나누기가 아님, 폴더 구조
        dir = new_base_dir / subset_name / category
        os.makedirs(dir)
        fnames = [f"{category}.{i}.jpg" for i in range(start_index, end_index)]
        for fname in fnames:
            shutil.copyfile(original_dir / fname, dir / fname)

# 아래 주석은 한 번만 실행
# make_subset("train", start_index=0, end_index=1000)
# make_subset("validation", start_index=1000, end_index=1500)
# make_subset("test", start_index=1500, end_index=2500)

# 데이터 증강
data_argumentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        # 10% 회전
        layers.RandomRotation(0.1),
        # 10% 확대
        layers.RandomZoom(0.1),
    ]
)
# 모델 구성

inputs = keras.Input(shape=(180, 180, 3))
# resize가 아닌 rescaling으로 0~255 사이의 값을 0~1 사이로 변환
x = data_argumentation(inputs)
x = layers.Rescaling(1./255)(x)
# x = layers.Rescaling(1./255)(inputs)
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.Flatten()(x)
# flatten 이후에 dropout 적용
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()

model.compile(loss="binary_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])



#180*180 크기로 이미지 크기 조정
train_dataset = image_dataset_from_directory(
    new_base_dir / "train",
    image_size=(180, 180),
    batch_size=32)
validation_dataset = image_dataset_from_directory(
    new_base_dir / "validation",
    image_size=(180, 180),
    batch_size=32)
test_dataset = image_dataset_from_directory(
    new_base_dir / "test",
    image_size=(180, 180),
    batch_size=32)


# 파이프라인 형태로 모델 저장
# callbacks = [
#     keras.callbacks.ModelCheckpoint(
#         # filepath="convnet_from_scratch.keras",
#         filepath="convnet_from_scratch_with_augumention.keras",
#         #best 모델만 저장(epoch 마다 저장X)
#         save_best_only=True,
#         monitor="val_loss")
# ]
# history = model.fit(
#     train_dataset,
#     epochs=100,
#     # epochs=30,
#     validation_data=validation_dataset,
#     callbacks=callbacks)

# accuracy = history.history["accuracy"]
# val_accuracy = history.history["val_accuracy"]
# loss = history.history["loss"]
# val_loss = history.history["val_loss"]
# epochs = range(1, len(accuracy) + 1)
# plt.plot(epochs, accuracy, "bo", label="Training accuracy")
# plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
# plt.title("Training and validation accuracy")
# plt.legend()
# plt.figure()
# plt.plot(epochs, loss, "bo", label="Training loss")
# plt.plot(epochs, val_loss, "b", label="Validation loss")
# plt.title("Training and validation loss")
# plt.legend()
# plt.show()

# test_model = keras.models.load_model("convnet_from_scratch.keras")
# test_loss, test_acc = test_model.evaluate(test_dataset)
# print(f"Test accuracy: {test_acc:.3f}")

test_model = keras.models.load_model("convnet_from_scratch_with_augumention.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")