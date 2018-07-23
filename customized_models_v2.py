import torch, torchvision
import torch.nn as nn
import torch.nn.init as init
import logging

class ModelCustomizer:

    def __init__(self, cnn_architecture_name):
        self.cnn_architecture_name = cnn_architecture_name
        self.available_models = {

        'resnet18': [torchvision.models.resnet18, 'fc'],
        'resnet34': [torchvision.models.resnet34, 'fc'],
        'resnet50': [torchvision.models.resnet50, 'fc'],
        'resnet152': [torchvision.models.resnet152, 'fc'],

        'densenet121': [torchvision.models.densenet121, 'classifier'],
        'densenet161': [torchvision.models.densenet161, 'classifier'],
        'densenet169': [torchvision.models.densenet169, 'classifier'],
        'densenet201': [torchvision.models.densenet201, 'classifier'],

        'vgg11_bn':[torchvision.models.vgg11_bn, 'classifier'],
        'vgg13_bn':[torchvision.models.vgg13_bn, 'classifier'],
        'vgg16_bn':[torchvision.models.vgg16_bn, 'classifier'],
        'vgg19_bn':[torchvision.models.vgg19_bn, 'classifier'],

        'inception_v3':[torchvision.models.inception_v3, 'fc']}

        assert cnn_architecture_name in self.available_models.keys(), 'Model {} is not available.'.format(cnn_architecture_name)

        self.cnn, self.layer_to_swap = self.available_models[cnn_architecture_name]

        try:
            self.cnn_architecture = self.cnn(pretrained = True)
        except TypeError:
            logging.warning(f'{cnn_architecture_name} does not support pretrain a keyword argument.')
            self.cnn_architecture = self.cnn()
        except:
            logging.warning('server problem encoutered. setting pretrained = False. This might affect the model performance.')
            self.cnn_architecture = self.cnn(pretrained = False)

    def network_modifier(self, out_features = 2):

        if self.layer_to_swap == 'fc':
            in_features  = self.cnn_architecture._modules[self.layer_to_swap].in_features

            self.cnn_architecture.fc = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.Sigmoid()
            )
            if self.cnn_architecture_name.startswith('inception'):
                logging.info('inception models take the image of size 299 x 299. Resize the input images accordingly.')
        else: #'classifier: densenet and vgg'
            if self.cnn_architecture_name.startswith('vgg'):
                out_feaures_ = self.cnn_architecture.classifier._modules['6'].out_features
                self.cnn_architecture.classifier._modules['7'] = nn.Linear(in_features= out_feaures_, out_features = out_features, bias=True)
                self.cnn_architecture.classifier._modules['8'] = nn.Sigmoid()

            else: #densenet
                in_features  = self.cnn_architecture._modules[self.layer_to_swap].in_features

                self.cnn_architecture.classifier = nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.Sigmoid()
                )

        return self.cnn_architecture

    def change_initialization(self):
        for module in self.cnn_architecture.modules():

            if isinstance(module, nn.Conv2d):
                init.xavier_uniform_(module.weight.data)
            elif isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight.data)
                if module.bias is not None:
                    init.constant_(module.bias, 0.0)

        return self.cnn_architecture

if __name__ == '__main__': #example

    logging.basicConfig(level = logging.INFO)
    m1 = ModelCustomizer('vgg16_bn')
    cnn = m1.network_modifier(14)
    cnn = m1.change_initialization()
    print(cnn)
