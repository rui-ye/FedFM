import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
try:
    from tensorflow.keras.applications import ResNet50
except:
    ResNet50 = None


class ModelFedCon_noheader(keras.Model):
    """Federated Learning model without header"""

    def __init__(self, base_model, out_dim, n_classes, net_configs=None):
        super(ModelFedCon_noheader, self).__init__()
        print('no header')
        
        if base_model == "resnet50":
            # ResNet50 with modified first layer for CIFAR-10
            with tf.device('/CPU:0'):
                if ResNet50:
                    base = ResNet50(include_top=False, weights=None, input_shape=(32, 32, 3))
                    # Build a new model with modified first layer
                    inputs = layers.Input(shape=(32, 32, 3))
                    x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', use_bias=False)(inputs)
                    x = layers.BatchNormalization()(x)
                    x = layers.ReLU()(x)
                    # Get features from ResNet50 (skip first few layers)
                    resnet_output = base(x)
                    x = layers.GlobalAveragePooling2D()(resnet_output)
                    self.features = keras.Model(inputs, x)
                else:
                    # Fallback: simple ResNet-like architecture
                    self.features = self._build_simple_resnet(50)
            num_ftrs = 2048
            
        elif base_model == "resnet50_7":
            with tf.device('/CPU:0'):
                if ResNet50:
                    base = ResNet50(include_top=False, weights=None, input_shape=(32, 32, 3))
                    x = base.output
                    x = layers.GlobalAveragePooling2D()(x)
                    self.features = keras.Model(base.input, x)
                else:
                    self.features = self._build_simple_resnet(50)
            num_ftrs = 2048
            
        elif base_model == "resnet18":
            # ResNet18 with modified first layer
            self.features = self._build_resnet18_cifar()
            num_ftrs = 512
            
        elif base_model == "resnet18_7":
            self.features = self._build_resnet18_cifar(use_7x7=False)
            num_ftrs = 512
            
        elif base_model == 'resnet18_7_gn':
            # ResNet18 with GroupNorm instead of BatchNorm
            self.features = self._build_resnet18_cifar_gn(use_7x7=False)
            num_ftrs = 512
            
        elif base_model == 'resnet18_gn':
            self.features = self._build_resnet18_cifar_gn(use_7x7=True)
            num_ftrs = 512
            
        elif base_model == 'resnet50_gn':
            # Use BatchNorm for now (GroupNorm can be added if needed)
            self.features = self._build_simple_resnet(50)
            num_ftrs = 2048
        else:
            raise ValueError(f"Unsupported base model: {base_model}")

        # Last layer
        self.l3 = layers.Dense(n_classes, name='classifier')

    def _build_resnet18_cifar(self, use_7x7=True):
        """Build ResNet18 for CIFAR-10"""
        # Force CPU for model construction to avoid GPU initialization issues
        with tf.device('/CPU:0'):
            def residual_block(x, filters, stride=1):
                shortcut = x
                if stride != 1 or x.shape[-1] != filters:
                    shortcut = layers.Conv2D(filters, 1, strides=stride, use_bias=False)(x)
                    shortcut = layers.BatchNormalization()(shortcut)
                
                x = layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False)(x)
                x = layers.BatchNormalization()(x)
                x = layers.ReLU()(x)
                x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
                x = layers.BatchNormalization()(x)
                x = layers.Add()([x, shortcut])
                x = layers.ReLU()(x)
                return x
            
            inputs = layers.Input(shape=(32, 32, 3))
            if use_7x7:
                x = layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False)(inputs)
            else:
                x = layers.Conv2D(64, 3, strides=1, padding='same', use_bias=False)(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            if use_7x7:
                x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
            
            x = residual_block(x, 64)
            x = residual_block(x, 64)
            x = residual_block(x, 128, stride=2)
            x = residual_block(x, 128)
            x = residual_block(x, 256, stride=2)
            x = residual_block(x, 256)
            x = residual_block(x, 512, stride=2)
            x = residual_block(x, 512)
            
            x = layers.GlobalAveragePooling2D()(x)
            return keras.Model(inputs, x)
    
    def _build_resnet18_cifar_gn(self, use_7x7=True):
        """Build ResNet18 with GroupNorm for CIFAR-10"""
        # Force CPU for model construction to avoid GPU initialization issues
        with tf.device('/CPU:0'):
            try:
                GroupNorm = layers.GroupNormalization
            except:
                # Fallback to BatchNorm if GroupNorm not available
                GroupNorm = layers.BatchNormalization
            
            def residual_block_gn(x, filters, stride=1):
                shortcut = x
                if stride != 1 or x.shape[-1] != filters:
                    shortcut = layers.Conv2D(filters, 1, strides=stride, use_bias=False)(x)
                    shortcut = GroupNorm(groups=2 if hasattr(GroupNorm, 'groups') else None)(shortcut)
                
                x = layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False)(x)
                x = GroupNorm(groups=2 if hasattr(GroupNorm, 'groups') else None)(x)
                x = layers.ReLU()(x)
                x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
                x = GroupNorm(groups=2 if hasattr(GroupNorm, 'groups') else None)(x)
                x = layers.Add()([x, shortcut])
                x = layers.ReLU()(x)
                return x
            
            inputs = layers.Input(shape=(32, 32, 3))
            if use_7x7:
                x = layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False)(inputs)
            else:
                x = layers.Conv2D(64, 3, strides=1, padding='same', use_bias=False)(inputs)
            x = GroupNorm(groups=2 if hasattr(GroupNorm, 'groups') else None)(x)
            x = layers.ReLU()(x)
            if use_7x7:
                x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
            
            x = residual_block_gn(x, 64)
            x = residual_block_gn(x, 64)
            x = residual_block_gn(x, 128, stride=2)
            x = residual_block_gn(x, 128)
            x = residual_block_gn(x, 256, stride=2)
            x = residual_block_gn(x, 256)
            x = residual_block_gn(x, 512, stride=2)
            x = residual_block_gn(x, 512)
            
            x = layers.GlobalAveragePooling2D()(x)
            return keras.Model(inputs, x)
    
    def _build_simple_resnet(self, depth=50):
        """Build a simple ResNet-like model"""
        # Force CPU for model construction to avoid GPU initialization issues
        with tf.device('/CPU:0'):
            inputs = layers.Input(shape=(32, 32, 3))
            x = layers.Conv2D(64, 3, padding='same', use_bias=False)(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            
            for _ in range(4):
                x = layers.Conv2D(64, 3, padding='same', use_bias=False)(x)
                x = layers.BatchNormalization()(x)
                x = layers.ReLU()(x)
            
            x = layers.GlobalAveragePooling2D()(x)
            return keras.Model(inputs, x)

    def call(self, x, training=None):
        h = self.features(x, training=training)
        # Ensure h is 2D
        if len(h.shape) > 2:
            h = tf.squeeze(h)
        # Handle batch size 1 case
        if len(h.shape) == 1:
            h = tf.expand_dims(h, 0)
        y = self.l3(h, training=training)
        return h, h, y
