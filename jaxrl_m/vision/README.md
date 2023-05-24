## jaxrl_m.vision

There are three main components to the vision module:

1. jaxrl_m.vision.encoders: A dictionary of common vision encoders (e.g. ResNet, Impala, VIT)
2. jaxrl_m.vision.data_augmentations: A module of common data augmentations (e.g. random crop, random flip, color jitter)
3. jaxrl_m.vision.pretrained_encoder: Useful preprocessing tools for interfacing with pre-trained models (see [dibyaghosh/pretrained_vision](https://github.com/dibyaghosh/pretrained_vision) for details on how to convert PyTorch checkpoints to JAX checkpoints)

### Supported Encoders

You can directly use a pre-configured encoder from the list below by calling `encoder_def = jaxrl_m.vision.encoders['impala']()`, or you can use the base files in this directory to create your own custom encoder.

| Type of Encoder | Most common variant | All versions                                                                                                    |
|-----------------|---------------------|-----------------------------------------------------------------------------------------------------------------|
| Impala          | impala              | impala, impala_large, impala_larger, impala_largest, impala_wider, impala_widest, impala_deeper, impala_deepest |
| ResnetV1        | resnetv1-50         | resnetv1-18, resnetv1-34, resnetv1-50, rensetv1-101, resnetv1-152, resnetv1-200                                 |
| ResnetV2        | resnetv2-50-1       | resnetv2-26-1, resnetv2-50-1                                                                                    |
| ViT             | ViT-B/16            | ViT-Ti/16, ViT-S/16, ViT-B/16, ViT-L/16                                                                         |
| MAE             | mae_base            | mae_debug, mase_small, mae_base, mae_large, mae_huge                                                            |
| Atari           | atari               | atari                                                                                                           |
| PTR Encoders    | resnetv1-34-bridge  | resnetv1-18-bridge, resnetv1-34-bridge                                                                          |
