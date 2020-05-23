from tensorflow.keras.layers import (
    ZeroPadding2D,
    BatchNormalization,
    LeakyReLU,
    Conv2D,
    Add,
    Input,
    UpSampling2D,
    Concatenate,
    Lambda,
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model
import tensorflow as tf
import numpy as np
import os
from Helpers.utils import get_boxes, timer, default_logger


class V3Model:
    def __init__(
        self,
        input_shape,
        classes=80,
        anchors=None,
        masks=None,
        max_boxes=100,
        iou_threshold=0.5,
        score_threshold=0.5,
    ):
        """
        Initialize yolov3 model.
        Args:
            input_shape: tuple(n, n, c)
            classes: Number of classes(defaults to 80 for Coco objects)
            anchors: numpy array of anchors (x, y) pairs
            masks: numpy array of masks.
            max_boxes: Maximum boxes in a single image.
            iou_threshold: Minimum overlap that counts as a valid detection.
            score_threshold: Minimum confidence that counts as a valid detection.
        """
        self.current_layer = 1
        self.input_shape = input_shape
        self.classes = classes
        self.anchors = anchors
        if anchors is None:
            self.anchors = np.array(
                [
                    (10, 13),
                    (16, 30),
                    (33, 23),
                    (30, 61),
                    (62, 45),
                    (59, 119),
                    (116, 90),
                    (156, 198),
                    (373, 326),
                ],
                np.float32,
            )
        if self.anchors[0][0] > 1:
            self.anchors = self.anchors / input_shape[0]
        self.masks = masks
        if masks is None:
            self.masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
        self.funcs = (
            ZeroPadding2D,
            BatchNormalization,
            LeakyReLU,
            Conv2D,
            Add,
            Input,
            UpSampling2D,
            Concatenate,
            Lambda,
            Model,
        )
        self.func_names = [
            'zero_padding',
            'batch_normalization',
            'leaky_relu',
            'conv2d',
            'add',
            'input',
            'up_sample',
            'concat',
            'lambda',
            'model',
        ]
        self.layer_names = {
            func.__name__: f'layer_CURRENT_LAYER_{name}'
            for func, name in zip(self.funcs, self.func_names)
        }
        self.shortcuts = []
        self.training_model = None
        self.inference_model = None
        self.output_layers = ['output_2', 'output_1', 'output_0']
        self.max_boxes = max_boxes
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

    def apply_func(self, func, x=None, *args, **kwargs):
        """
        Apply a function from self.funcs and increment layer count.
        Args:
            func: func from self.funcs.
            x: image tensor.
            *args: func args
            **kwargs: func kwargs

        Returns:
            result of func
        """
        name = self.layer_names[func.__name__].replace(
            'CURRENT_LAYER', f'{self.current_layer}'
        )
        if func.__name__ == 'Model':
            name = f'layer_{self.current_layer}_{self.output_layers.pop()}'
        result = func(name=name, *args, **kwargs)
        self.current_layer += 1
        if x is not None:
            return result(x)
        return result

    def convolution_block(
        self, x, filters, kernel_size, strides, batch_norm, action=None
    ):
        """
        Convolution block for yolo version3.
        Args:
            x: Image input tensor.
            filters: Number of filters/kernels.
            kernel_size: Size of the filter/kernel.
            strides: The number of pixels a filter moves, like a sliding window.
            batch_norm: Standardizes the inputs to a layer for each mini-batch.
            action: 'add' or 'append'

        Returns:
            x or x added to shortcut.
        """
        if action == 'append':
            self.shortcuts.append(x)
        padding = 'same'
        if strides != 1:
            x = self.apply_func(ZeroPadding2D, x, padding=((1, 0), (1, 0)))
            padding = 'valid'
        x = self.apply_func(
            Conv2D,
            x,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=not batch_norm,
            kernel_regularizer=l2(0.0005),
        )
        if batch_norm:
            x = self.apply_func(BatchNormalization, x)
            x = self.apply_func(LeakyReLU, x, alpha=0.1)
        if action == 'add':
            return self.apply_func(Add, [self.shortcuts.pop(), x])
        return x

    def output(self, x_input, filters):
        """
        Output layer.
        Args:
            x_input: image tensor.
            filters: number of convolution filters.

        Returns:
            tf.keras.models.Model
        """
        x = inputs = self.apply_func(Input, shape=x_input.shape[1:])
        x = self.convolution_block(x, 2 * filters, 3, 1, True)
        x = self.convolution_block(x, 3 * (5 + self.classes), 1, 1, False)
        x = self.apply_func(
            Lambda,
            x,
            lambda item: tf.reshape(
                item,
                (
                    -1,
                    tf.shape(item)[1],
                    tf.shape(item)[2],
                    3,
                    self.classes + 5,
                ),
            ),
        )
        return self.apply_func(Model, x_input, inputs, x)

    def get_nms(self, outputs):
        """
        Apply non-max suppression and get valid detections.
        Args:
            outputs: yolov3 model outputs.

        Returns:
            boxes, scores, classes, valid_detections
        """
        boxes, conf, type_ = [], [], []
        for output in outputs:
            boxes.append(
                tf.reshape(
                    output[0],
                    (tf.shape(output[0])[0], -1, tf.shape(output[0])[-1]),
                )
            )
            conf.append(
                tf.reshape(
                    output[1],
                    (tf.shape(output[1])[0], -1, tf.shape(output[1])[-1]),
                )
            )
            type_.append(
                tf.reshape(
                    output[2],
                    (tf.shape(output[2])[0], -1, tf.shape(output[2])[-1]),
                )
            )
        bbox = tf.concat(boxes, axis=1)
        confidence = tf.concat(conf, axis=1)
        class_probabilities = tf.concat(type_, axis=1)
        scores = confidence * class_probabilities
        (
            boxes,
            scores,
            classes,
            valid_detections,
        ) = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(bbox, (tf.shape(bbox)[0], -1, 1, 4)),
            scores=tf.reshape(
                scores, (tf.shape(scores)[0], -1, tf.shape(scores)[-1])
            ),
            max_output_size_per_class=self.max_boxes,
            max_total_size=self.max_boxes,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
        )
        return boxes, scores, classes, valid_detections

    def create_layer(self,
                     layer_configuration,
                     x,
                     skips,
                     detections,
                     training_outputs,
                     concats,
                     input_initial,
                     inference_outputs):
        if 'conv' in layer_configuration:
            if len(layer_configuration) < 6:
                layer_configuration = (
                        [int(item) for item in layer_configuration[1: 4]] +
                        ([bool(layer_configuration[4])]))
            else:
                layer_configuration = (
                        [int(item) for item in layer_configuration[1: 4]] +
                        ([bool(layer_configuration[4])] + [layer_configuration[5]]))
            return self.convolution_block(x, *layer_configuration)
        if 'skip' in layer_configuration[0]:
            if skips:
                skips['skip_61'] = x
            else:
                skips['skip_36'] = x
        if 'detection' in layer_configuration[0]:
            detections.append(x)
        if 'output' in layer_configuration[0]:
            out = self.output(detections.pop(), int(layer_configuration[1]))
            training_outputs.append(out)
        if 'upsample' in layer_configuration:
            return self.apply_func(UpSampling2D, x, size=2)
        if 'concat' in layer_configuration:
            if concats:
                result = self.apply_func(Concatenate, [x, skips['skip_36']])
            else:
                result = self.apply_func(Concatenate, [x, skips['skip_61']])
                concats.append(1)
            return result
        if 'training_model' in layer_configuration:
            self.training_model = Model(
                input_initial,
                training_outputs,
                name='training_model',
            )
        if 'boxes' in layer_configuration[0]:
            box_index = int(layer_configuration[0].split('_')[-1])
            result = self.apply_func(
                Lambda,
                training_outputs[box_index],
                lambda item: get_boxes(
                    item, self.anchors[self.masks[box_index]], self.classes
                ),
            )
            inference_outputs.append(result)
        if 'nms' in layer_configuration:
            inference_outputs = self.apply_func(
                Lambda,
                (inference_outputs[0][:3],
                 inference_outputs[1][:3],
                 inference_outputs[2][:3]),
                lambda item: self.get_nms(item),
            )
        if 'inference_model' in layer_configuration:
            self.inference_model = Model(
                input_initial, inference_outputs, name='inference_model'
            )

    @timer(default_logger)
    def create_models(self, configuration):
        """
        Create training and inference yolov3 models.
        Args:
            configuration: Yolo layer configuration file.

        Returns:
            training, inference models
        """
        input_initial = self.apply_func(Input, shape=self.input_shape)
        x = input_initial
        skips, output_layers, detection_layers, training_outs, inference_outs, concats = (
            {}, [], [], [], [], [])
        layers = [item.strip() for item in open(configuration).readlines()]
        layers = list(map(lambda l: l.split(',') if ',' in l else [l], layers))
        for layer in layers:
            result = self.create_layer(
                layer, x, skips, detection_layers, training_outs,
                concats, input_initial, inference_outs)
            if result is not None:
                x = result
        default_logger.info('Training and inference models created')
        return self.training_model, self.inference_model

    @timer(default_logger)
    def load_weights(self, weights_file):
        """
        Load DarkNet weights or checkpoint/pre-trained weights.
        Args:
            weights_file: .weights or .tf file path.

        Returns:
            None
        """
        assert weights_file.split('.')[-1] in [
            'tf',
            'weights',
        ], 'Invalid weights file'
        assert (
            self.classes == 80 if weights_file.endswith('.weights') else 1
        ), f'DarkNet model should contain 80 classes, {self.classes} is given.'
        if weights_file.endswith('.tf'):
            self.training_model.load_weights(weights_file)
            default_logger.info(f'Loaded weights: {weights_file} ... success')
            return
        with open(weights_file, 'rb') as weights_data:
            default_logger.info(f'Loading pre-trained weights ...')
            major, minor, revision, seen, _ = np.fromfile(
                weights_data, dtype=np.int32, count=5
            )
            all_layers = [
                layer
                for layer in self.training_model.layers
                if 'output' not in layer.name
            ]
            output_models = [
                layer
                for layer in self.training_model.layers
                if 'output' in layer.name
            ]
            output_layers = [item.layers for item in output_models]
            for output_item in output_layers:
                all_layers.extend(output_item)
            all_layers.sort(key=lambda layer: int(layer.name.split('_')[1]))
            for i, layer in enumerate(all_layers):
                current_read = weights_data.tell()
                total_size = os.fstat(weights_data.fileno()).st_size
                print(
                    f'\r{round(100 * (current_read / total_size))}%\t{current_read}/{total_size}',
                    end='',
                )
                if 'conv2d' not in layer.name:
                    continue
                next_layer = all_layers[i + 1]
                b_norm_layer = (
                    next_layer
                    if 'batch_normalization' in next_layer.name
                    else None
                )
                filters = layer.filters
                kernel_size = layer.kernel_size[0]
                input_dimension = layer.get_input_shape_at(-1)[-1]
                convolution_bias = (
                    np.fromfile(weights_data, dtype=np.float32, count=filters)
                    if b_norm_layer is None
                    else None
                )
                bn_weights = (
                    np.fromfile(
                        weights_data, dtype=np.float32, count=4 * filters
                    ).reshape((4, filters))[[1, 0, 2, 3]]
                    if (b_norm_layer is not None)
                    else None
                )
                convolution_shape = (
                    filters,
                    input_dimension,
                    kernel_size,
                    kernel_size,
                )
                convolution_weights = (
                    np.fromfile(
                        weights_data,
                        dtype=np.float32,
                        count=np.product(convolution_shape),
                    )
                    .reshape(convolution_shape)
                    .transpose([2, 3, 1, 0])
                )
                if b_norm_layer is None:
                    try:
                        layer.set_weights(
                            [convolution_weights, convolution_bias]
                        )
                    except ValueError:
                        pass
                if b_norm_layer is not None:
                    layer.set_weights([convolution_weights])
                    b_norm_layer.set_weights(bn_weights)
            assert len(weights_data.read()) == 0, 'failed to read all data'
        default_logger.info(f'Loaded weights: {weights_file} ... success')
        print()


if __name__ == '__main__':
    mod = V3Model((416, 416, 3), 80)
    tr, inf = mod.create_models('../Config/yolo3_3o.txt')
    mod.load_weights('../../../yolov3.weights')
