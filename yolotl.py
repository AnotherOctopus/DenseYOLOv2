from keras import backend as K
from keras.layers import Dense, Conv2D, Activation, LeakyReLU, MaxPooling2D
from keras.optimizers import Adam
from keras.models import Sequential
model = Sequential()

model.add(Conv2D(
    filters=64,
    kernel_size= (7,7),
    strides=2,
    padding="valid",
    data_format="channels_last",
    use_bias=False,
    kernel_initializer='random_uniform'
    ))

model.add(MaxPooling2D(
   pool_size=(2,2),
   strides=(2,2),
   padding="valid",
   data_format="channels_last"
   ))
model.add(LeakyReLU(alpha=0.3))


optimizer = Adam(
        lr=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=None,
        decay=0.0001,
        amsgrad = False
        )

