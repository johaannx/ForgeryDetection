import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Input,UpSampling2D
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import gc

# Memory optimization settings
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Enable mixed precision
tf.keras.mixed_precision.set_global_policy('float32')


def load_comofod_data(base_dir, img_size=(512, 512)):
    """
    Load CoMoFoD dataset images and masks with memory-efficient processing
    """
    images = []
    masks = []
    
    original_dir = os.path.join(base_dir, 'original')
    forged_dir = os.path.join(base_dir, 'forged')
    mask_dir = os.path.join(base_dir, 'mask')
    
    for img_name in sorted(os.listdir(original_dir)):
        if img_name.endswith('_O.png'):
            base_name = img_name.split('_')[0]
            original_path = os.path.join(original_dir, f"{base_name}_O.png")
            forged_path = os.path.join(forged_dir, f"{base_name}_F.png")
            mask_path = os.path.join(mask_dir, f"{base_name}_B.png")
            
            if all(os.path.exists(p) for p in [original_path, forged_path, mask_path]):
                original = cv2.imread(original_path)
                original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
                original = cv2.resize(original, img_size) / 255.0
                
                forged = cv2.imread(forged_path)
                forged = cv2.cvtColor(forged, cv2.COLOR_BGR2RGB)
                forged = cv2.resize(forged, img_size) / 255.0
                
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, img_size)
                mask = (mask > 0).astype(np.float32)
                
                images.extend([original, forged])
                masks.extend([np.zeros_like(mask), mask])
                
                del original, forged, mask
                if len(images) % 20 == 0:
                    gc.collect()
            else:
                print(f"Warning: Incomplete set for {base_name}")
    
    return np.array(images, dtype='float16'), np.array(masks, dtype='float16')[..., np.newaxis]

def create_unet_model(input_size=(512, 512, 3)):
    """Define memory-efficient U-Net architecture"""
    inputs = Input(input_size)
    
    # Encoder
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)
    
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    
    # Bridge
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(c5)
    

    # Decoder
    u6 = UpSampling2D((2, 2))(c5)
    u6 = Conv2D(256, (3, 3), activation='relu', padding='same')(u6)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same')(c6)

    u7 = UpSampling2D((2, 2))(c6)
    u7 = Conv2D(128, (3, 3), activation='relu', padding='same')(u7)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same')(c7)

    u8 = UpSampling2D((2, 2))(c7)
    u8 = Conv2D(64, (3, 3), activation='relu', padding='same')(u8)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same')(c8)

    u9 = UpSampling2D((2, 2))(c8)
    u9 = Conv2D(32, (3, 3), activation='relu', padding='same')(u9)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(32, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(32, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model


def plot_results(original, forged, true_mask, pred_mask, index):
    """Plot and save results with input arrays cast to float32 for compatibility"""
    plt.figure(figsize=(15, 5))
    
    # Ensure all images are float32 for compatibility
    original = original.astype(np.float32)
    forged = forged.astype(np.float32)
    true_mask = true_mask.astype(np.float32)
    pred_mask = pred_mask.astype(np.float32)
    
    plt.subplot(141)
    plt.title('Original Image')
    plt.imshow(original)
    plt.axis('off')
    
    plt.subplot(142)
    plt.title('Forged Image')
    plt.imshow(forged)
    plt.axis('off')
    
    plt.subplot(143)
    plt.title('True Mask')
    plt.imshow(true_mask[:, :, 0], cmap='gray')
    plt.axis('off')
    
    plt.subplot(144)
    plt.title('Predicted Mask')
    plt.imshow(pred_mask[:, :, 0], cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join('results', f'result_{int(index)}.png'))
    plt.close()


def main():
    # Configuration
    DATASET_DIR = "organised_dataset/"
    IMG_SIZE = (512, 512)
    BATCH_SIZE = 4  # Reduced batch size
    EPOCHS = 50
    VALIDATION_SPLIT = 0.2
    
    # Clear any existing models/memory
    tf.keras.backend.clear_session()
    gc.collect()
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    images, masks = load_comofod_data(DATASET_DIR, IMG_SIZE)
    
    # Split dataset
    train_images, val_images, train_masks, val_masks = train_test_split(
        images, masks, test_size=VALIDATION_SPLIT, random_state=42
    )
    
    # Free up original arrays
    del images, masks
    gc.collect()
    
    print(f"Training samples: {len(train_images)}")
    print(f"Validation samples: {len(val_images)}")
    
    # Create and compile model
    print("Creating model...")
    model = create_unet_model(input_size=(*IMG_SIZE, 3))
    
    # Use memory-efficient optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    # Create callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            save_best_only=True,
            monitor='val_loss',
            save_freq='epoch'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.CSVLogger('training_history.csv')
    ]
    
    # Train model
    print("Training model...")
    try:
        history = model.fit(
            train_images,
            train_masks,
            validation_data=(val_images, val_masks),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1 # Set to True if you use data generators
        )
    except Exception as e:
        print(f"Error during training: {e}")
    
    # Save the final model after training
    model.save('final_model.h5')
    print("Model saved as 'final_model.h5'.")

    # Evaluate and visualize results
    print("Evaluating model...")
    pred_masks = model.predict(val_images, batch_size=BATCH_SIZE)
    for i in range(5):  # Change this for more visualizations
        plot_results(val_images[i], val_images[i], val_masks[i], pred_masks[i], float(i))

if __name__ == "__main__":
    main()
