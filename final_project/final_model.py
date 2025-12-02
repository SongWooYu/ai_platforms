import os, shutil, pathlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory

# ==========================================
# 1. 데이터 필터링 (폴더 복사 수행)
# ==========================================
original_dir = pathlib.Path("MNIST300")      # 원본 데이터
new_base_dir = pathlib.Path("mnist_filtered") # 복사할 위치
target_numbers = ["2", "4", "8"]             # 골라낼 숫자

def setup_filtered_dataset():
    # 기존에 만들어진 폴더가 있다면 삭제하고 깨끗하게 다시 시작
    if os.path.exists(new_base_dir):
        shutil.rmtree(new_base_dir)
    
    print(f"데이터 복사 시작... ({original_dir} -> {new_base_dir})")
    
    for subset in ["training", "testing"]:
        for category in target_numbers:
            src_dir = original_dir / subset / category
            dst_dir = new_base_dir / subset / category
            
            if not os.path.exists(src_dir):
                print(f"[경고] 원본 경로 없음: {src_dir}")
                continue
                
            os.makedirs(dst_dir)
            
            # 파일 하나하나 복사
            fnames = os.listdir(src_dir)
            for fname in fnames:
                shutil.copyfile(src_dir / fname, dst_dir / fname)
                
    print("데이터 필터링 및 복사 완료!")

# 필터링 실행
setup_filtered_dataset()

# ==========================================
# 2. 데이터 로드 및 리사이징
# ==========================================
BATCH_SIZE = 32
IMG_SIZE = (224, 224) # VGG16 입력 크기

print("\n데이터셋 로드 중...")
# 복사된 폴더(mnist_filtered)에서 로드합니다.
train_dataset = image_dataset_from_directory(
    new_base_dir / "training",
    image_size=IMG_SIZE,    # [질문하신 부분] 여기서 300->224로 자동 변환됩니다.
    batch_size=BATCH_SIZE,
    color_mode='rgb',       
    shuffle=True
)

test_dataset = image_dataset_from_directory(
    new_base_dir / "testing",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='rgb'
)

# [전처리] VGG16 전용 전처리 함수 적용 (스케일링 포함)
def preprocess(images, labels):
    return keras.applications.vgg16.preprocess_input(images), labels

train_dataset = train_dataset.map(preprocess)
test_dataset = test_dataset.map(preprocess)

# ==========================================
# 3. 모델 정의 (VGG16 전이학습)
# ==========================================
conv_base = keras.applications.vgg16.VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
conv_base.trainable = False

inputs = keras.Input(shape=(224, 224, 3))
x = conv_base(inputs)
x = layers.Flatten()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(3, activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.summary()

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])

# ==========================================
# 4. 학습 및 저장
# ==========================================
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="vgg16_mnist_248_best.h5", 
        save_best_only=True,
        monitor="val_loss")
]

history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=test_dataset,
    callbacks=callbacks
)

# ==========================================
# 5. 결과 그래프
# ==========================================
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, "bo", label="Training accuracy")
plt.plot(epochs, val_acc, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()

# ==========================================
# 6. 최종 성능 평가
# ==========================================
best_model = keras.models.load_model("vgg16_mnist_248_best.h5")
test_loss, test_acc = best_model.evaluate(test_dataset)
print("-" * 30)
print(f"Testing 데이터셋 최고 정확도: {test_acc:.4f}")
print("-" * 30)