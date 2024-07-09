import tensorflow as tf

from tcav.model import ModelWrapper
from tensorflow.keras.utils import to_categorical
from extractdd import BN_LAYER


class KerasModelWrapper(ModelWrapper):
    """ ModelWrapper for keras models

    By default, assumes that your model contains one input node, one output head
    and one loss function.
    Computes gradients of the output layer in respect to a CAV.

    Args:
        sess: Tensorflow session we will use for TCAV.
        model_path: Path to your model.h5 file, containing a saved trained
          model.
        labels_path: Path to a file containing the labels for your problem. It
          requires a .txt file, where every line contains a label for your
          model. You want to make sure that the order of labels in this file
          matches with the logits layers for your model, such that file[i] ==
          model_logits[i]
  """

    def __init__(
            self,
            sess,
            model_path,
            labels_path,
    ):
        self.sess = sess
        super(KerasModelWrapper, self).__init__()
        self.import_keras_model(model_path)
        self.labels = tf.io.gfile.GFile(labels_path).read().splitlines()

        # Construct gradient ops. Defaults to using the model's output layer
        self.y_input = tf.compat.v1.placeholder(tf.int64, shape=[None, 1000])
        self.loss = self.model.loss_functions[0](self.y_input,
                                                 self.model.outputs[0])
        self._make_gradient_tensors()

    def id_to_label(self, idx):
        return self.labels[idx]

    def label_to_id(self, label):
        return self.labels.index(label)

    def import_keras_model(self, saved_path):
        """Loads keras model, fetching bottlenecks, inputs and outputs."""
        self.ends = {}
        self.model = tf.keras.models.load_model(saved_path)
        self.get_bottleneck_tensors()
        self.get_inputs_and_outputs_and_ends()

    def get_bottleneck_tensors(self):
        self.bottlenecks_tensors = {}
        layers = self.model.layers
        for layer in layers:
            if 'input' not in layer.name:
                self.bottlenecks_tensors[layer.name] = layer.output

    def get_inputs_and_outputs_and_ends(self):
        self.ends['input'] = self.model.inputs[0]
        self.ends['prediction'] = self.model.outputs[0]


class ResNet50Wrapper_public(KerasModelWrapper):
    def __init__(self, sess, model_saved_path, labels_path):
        self.image_value_range = (-1, 1)
        self.sess = sess

        super(ResNet50Wrapper_public, self).__init__(
            sess,
            model_saved_path,
            labels_path)
        self.model_name = 'resnet50_bn_pretrained'

    def get_image_shape(self):
        return (224, 224, 3)

    def run_examples(self, images, BOTTLENECK_LAYER):
        return self.sess.run(self.bottlenecks_tensors[BOTTLENECK_LAYER],
                             {self.ends['input']: images})

    # def label_to_id(self, CLASS_NAME):
    #     labels = {}
    #     with open(self.label_txt, 'r') as f:
    #         line = f.readline()
    #         while line:
    #             label = line.split('\'')[1]
    #             if ',' in label:
    #                 label = label.split(',')[0]
    #             labels[label] = int(line.split(':')[0])
    #             line = f.readline()
    #         f.close()
    #     return labels[CLASS_NAME]

    def get_gradient(self, activations, CLASS_ID, BOTTLENECK_LAYER):
        return self.sess.run(self.bottlenecks_gradients[BOTTLENECK_LAYER], {
            self.bottlenecks_tensors[BOTTLENECK_LAYER]: activations,
            self.y_input: to_categorical(CLASS_ID, 1000)
        })
    def _make_gradient_tensors(self):
        """Makes gradient tensors for all bottleneck tensors."""
        
        self.bottlenecks_gradients = {}
        for bn in self.bottlenecks_tensors:
            if bn == BN_LAYER:
                self.bottlenecks_gradients[bn] = tf.gradients(
                    ys=self.loss, xs=self.bottlenecks_tensors[bn])[0]
        
