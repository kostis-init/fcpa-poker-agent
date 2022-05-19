import keras

model = keras.models.load_model('models/model_0')
model.save('models/model_0.h5')

model = keras.models.load_model('models/model_1')
model.save('models/model_1.h5')

model = keras.models.load_model('models/model_2')
model.save('models/model_2.h5')

model = keras.models.load_model('models/model_3')
model.save('models/model_3.h5')

model = keras.models.load_model('models/model_actions')
model.save('models/model_actions.h5')
