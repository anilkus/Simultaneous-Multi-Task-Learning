import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten,concatenate, Reshape
# Hazır veri seti oluşturma (örnek)
# Bu veri seti örnek amaçlıdır, gerçek bir veri setiyle değiştirilmelidir.
# Sınıflandırma için öznitelikler
classification_features = np.random.rand(1000, 10) # 1000 örnek, 10 öznitelik
# Sınıflandırma için etiketler
classification_labels = np.random.randint(2, size=(1000, 1)) # 0 veya 1 etiketleri

# Segmentasyon için veri
segmentation_data = np.random.rand(1000, 32, 32, 3) # 1000 örnek, 32x32 boyutunda,3 kanallı görüntüler
# Regresyon için veri
regression_data = np.random.rand(1000, 5) # 1000 örnek, 5 öznitelik
# Giriş katmanları
classification_input = Input(shape=(10,))
segmentation_input = Input(shape=(32, 32, 3))
regression_input = Input(shape=(5,))
# Sınıflandırma modeli
x1 = Dense(64, activation='relu')(classification_input)
x1 = Dense(32, activation='relu')(x1)
classification_output = Dense(1, activation='sigmoid')(x1)
# Segmentasyon modeli
x2 = Conv2D(32, (3, 3), activation='relu')(segmentation_input)
x2 = MaxPooling2D((2, 2))(x2)
x2 = Conv2D(64, (3, 3), activation='relu')(x2)
x2 = MaxPooling2D((2, 2))(x2)
x2 = Flatten()(x2)
segmentation_output = Dense(32*32*3, activation='sigmoid')(x2) # Tüm pikseller için bir olasılık değeri çıkarttık
segmentation_output = Reshape((32, 32, 3))(segmentation_output) # Çıktıyı giriş görüntülerinin boyutlarına çevirdik
# Regresyon modeli
x3 = Dense(64, activation='relu')(regression_input)
x3 = Dense(32, activation='relu')(x3)
regression_output = Dense(1, activation='linear')(x3)

# Model birleştirme aşaması..
combined_model = Model(inputs=[classification_input, segmentation_input,
regression_input],
 outputs=[classification_output, segmentation_output, regression_output])
# Model derlendi..
combined_model.compile(optimizer='adam',
 loss=['binary_crossentropy', 'binary_crossentropy', 'mse'],
 metrics=['accuracy'])
# Modeli eğitme
combined_model.fit([classification_features, segmentation_data, regression_data],
 [classification_labels, segmentation_data, regression_data],
 epochs=10, batch_size=64)

