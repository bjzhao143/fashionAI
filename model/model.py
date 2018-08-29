from keras.layers import *
from keras.applications import *
from keras.models import *

def Dual_channel_multitask(width,subtract_mean=[123, 117, 104],
                           divide_by_stddev=None):

    def input_mean_normalization(tensor):
        return tensor - np.array(subtract_mean)

    def input_stddev_normalization(tensor):
        return tensor / np.array(divide_by_stddev)

    model_name = 'dual_channel'
    base_model1 = ResNet50(include_top=False, weights='imagenet', input_shape=(width, width, 3), pooling='max')
    base_model2 = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=(width, width, 3), pooling='max')

    x = Input(shape=(height, width, 3))
    if not (subtract_mean is None):
        x = Lambda(input_mean_normalization, output_shape=(img_height, img_width, img_channels),
                   name='input_mean_normalization')(x)
    if not (divide_by_stddev is None):
        xx = Lambda(input_stddev_normalization, output_shape=(img_height, img_width, img_channels),
                    name='input_stddev_normalization')(x)

    x1 = Lambda(resnet50.preprocess_input)(x)
    x1 = base_model(x1)
    x1 = Dropout(0.5)(x1)

    x2 = Lambda(inception_resnet_v2.preprocess_input(x))
    x2 = base_model2(x2)
    x2 = Dropout(0.5)(x2)

    x = concatenate([x1,x2])
    x = [Dense(count, activation='softmax', name=name)(x) for name, count in label_count.items()]

    model = Model(input_tensor, x)
    return model
