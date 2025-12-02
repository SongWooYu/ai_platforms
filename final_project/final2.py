import os, shutil, pathlib
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory

# ==========================================
# 1. 데이터셋 재구성 (2, 4, 8 클래스만 추출)
# ==========================================

# 원본 데이터 경로 (압축 푼 MNIST300 폴더 위치에 맞게 수정하세요)
# 예: 현재 폴더 상위에 있다면 "../MNIST300"
original_dir = pathlib.Path("./MNIST300") 

# 새로 만들 데이터셋 경로
new_base_dir = pathlib.Path("./mnist_subset_248")

def make_subset_specific_classes(subset_name):
    target_classes = ["2", "4", "8"] # [요구사항] 2, 4, 8 숫자만 학습
    
    for category in target_classes:
        # 원본 경로: ../MNIST300/training/2
        src_dir = original_dir / subset_name / category
        # 새 경로: ./mnist_subset_248/training/2
        dest_dir = new_base_dir / subset_name / category
        
        # 디렉토리가 없으면 생성하고 파일 복사
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
            # 해당 폴더의 모든 파일 복사
            if os.path.exists(src_dir):
                fnames = os.listdir(src_dir)
                for fname in fnames:
                    shutil.copyfile(src_dir / fname, dest_dir / fname)

# 폴더가 이미 존재하지 않을 때만 실행 (데이터 중복 복사 방지)
if not os.path.exists(new_base_dir):
    print("데이터셋 재구성을 시작합니다...")
    make_subset_specific_classes("training")
    make_subset_specific_classes("testing")
    print("데이터셋 재구성 완료.")
else:
    print("이미 데이터셋이 존재합니다. 생성을 건너뜁니다.")


# ==========================================
# 2. 데이터 로드 (전처리 및 리사이징)
# ==========================================

BATCH_SIZE = 32
IMG_SIZE = (224, 224) # [요구사항] VGG16 권장 사이즈로 변경

print("데이터 로드 중...")
# 학습 데이터셋
train_dataset = image_dataset_from_directory(
    new_base_dir / "training",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical', # [중요] 3개 클래스 분류이므로 categorical 사용
    shuffle=True
)

# 테스트 데이터셋 (검증용)
test_dataset = image_dataset_from_directory(
    new_base_dir / "testing",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# ==========================================
# 3. 모델 정의 (VGG16 전이학습)
# ==========================================

# VGG16 모델 불러오기 (ImageNet 가중치, Top 제외)
conv_base = keras.applications.vgg16.VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
conv_base.trainable = False # 가중치 동결

# 데이터 증강 (숫자 이미지 특성상 좌우 반전은 제외하거나 주의 필요)
data_augmentation = keras.Sequential(
    [
        layers.RandomRotation(0.1), # 10% 회전
        layers.RandomZoom(0.1),     # 10% 확대
    ]
)

inputs = keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)

# VGG16 전용 전처리 (0~1 스케일링이 아닌, VGG 방식의 정규화 수행)
x = keras.applications.vgg16.preprocess_input(x)

x = conv_base(x)
x = layers.Flatten()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)

# [중요] 출력층 변경: 클래스 3개 (2,4,8), 활성화 함수 Softmax
outputs = layers.Dense(3, activation="softmax")(x)

model = keras.Model(inputs, outputs)

# 모델 컴파일
model.compile(loss="categorical_crossentropy", # 다중 분류 손실 함수
              optimizer="rmsprop",
              metrics=["accuracy"])

model.summary()

# ==========================================
# 4. 모델 학습
# ==========================================

# 체크포인트 설정
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="mnist_248_vgg16.h5",
        save_best_only=True,
        monitor="val_loss")
]

print("모델 학습 시작...")
history = model.fit(
    train_dataset,
    epochs=10, # 테스트를 위해 10회 설정 (필요시 20~30으로 증가)
    validation_data=test_dataset,
    callbacks=callbacks
)

# ==========================================
# 5. 결과 시각화 (그래프)
# ==========================================

accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)

# 정확도 그래프
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, accuracy, "bo", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()

# 손실 그래프
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()

plt.show()

# ==========================================
# 6. 최종 모델 평가
# ==========================================
print("최적 모델 로드 및 최종 평가:")
test_model = keras.models.load_model("mnist_248_vgg16.h5")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}")