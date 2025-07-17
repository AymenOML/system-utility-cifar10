from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_model(model, x_train, y_train, x_test, y_test, batch_size=64, epochs=20):
    datagen = ImageDataGenerator(
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        rotation_range=20
    )
    datagen.fit(x_train)

    steps = int(x_train.shape[0] / batch_size)
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        epochs=epochs,
        steps_per_epoch=steps,
        validation_data=(x_test, y_test)
    )
    return history
