from jaxrl_m.vision.impala import impala_configs
from jaxrl_m.vision.resnet_v2 import resnetv2_configs
from jaxrl_m.vision.small_encoders import small_configs
from jaxrl_m.vision.bridge_resnet_v1 import bridge_resnetv1_configs
from jaxrl_m.vision.resnet_v1 import vanilla_resnetv1_configs

from jaxrl_m.vision.mae import mae_model_configs
from jaxrl_m.vision.vit import vit_configs

from jaxrl_m.vision import data_augmentations
from jaxrl_m.vision.preprocess import PreprocessEncoder, BasicPreprocessEncoder

encoders = dict()
encoders.update(impala_configs)
encoders.update(resnetv2_configs)
encoders.update(bridge_resnetv1_configs)
encoders.update(vanilla_resnetv1_configs)
encoders.update(small_configs)

encoders.update(mae_model_configs)
encoders.update(vit_configs)