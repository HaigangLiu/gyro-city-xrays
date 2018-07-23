import cv2
from copy import deepcopy
from torch.nn import functional as F
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from data_utilities import ChestXrayDataSetMultiLabel, SpecializedContructorForCAM
from torchvision import transforms
from torch.utils.data import DataLoader
BATCH_SIZE = 1
class CamGenerator():
    """
    CAM: Class Activation Map for multi-classes
    Generate a heatmap on top of the existing image
    To help visualize the convolutional nueral network.
    For more details please see Zhou et. al (2016)


    Args:
        model: pytorch model object, a trained model
        name_of_the_last_conv_layer (str): The name of the layer to visualize
    """

    class_dict = { 'Atelectasis': 0, 'Cardiomegaly': 1, 'Effusion':2 , 'Infiltration':3, 'Mass':4, 'Nodule':5, 'Pneumonia':6,'Pneumothorax':7, 'Consolidation':8, 'Edema':9, 'Emphysema':10, 'Fibrosis':11, 'Pleural_Thickening':12, 'Hernia:2':13}

    def __init__(self, model, name_of_the_last_conv_layer, specialized_dataloader):

        self.model = model
        self.name_of_the_last_conv_layer = name_of_the_last_conv_layer

        self.feature_maps = self.gradient = None
        self.specialized_dataloader = specialized_dataloader
        self._register()

    def _register(self):

        def forward_recorder(module, input, output):
            self.feature_maps = output.data.cpu()

        def backward_recorder(module, grad_in, grad_out):
            self.gradient = grad_out[0].data.cpu()

        for i, j in self.model.named_modules():
            if i ==  self.name_of_the_last_conv_layer:
                try:
                    self.length_of_filter = j.out_channels
                    self.size_of_kernel = j.kernel_size
                except AttributeError:
                    print("Target layer is not conv2d layer")

                j.register_forward_hook(forward_recorder)
                j.register_backward_hook(backward_recorder)
                break
        else:
            print("Cannot find the given layer, maybe try another name.")
            return None

    def load_data(self, data_loader, which_class):
        try:# need a specialized dataloader
            self.image_tensor, self.label = next(iter(data_loader))
        except:
            raise ValueError('Need to pass a PyTorch DataLoader object as the positional argument')

        # assert len(self.label) == 1, 'Set the batch size of dataloader to 1'
        self.image_numpy = self.image_tensor[0].numpy().transpose(1,2,0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        self.image_numpy = std * self.image_numpy + mean
        self._forward_pass()
        self._backward_pass(which_class)

    def _forward_pass(self):
        self.output = self.model(self.image_tensor)
       # self.size_of_feature_maps = self.feature_maps.size()[2:]
        batch_size, self.channels, self.size_of_feature_maps, _ = self.feature_maps.size()
        try:
            self.feature_maps = self.feature_maps.squeeze(0)
        except:
            raise ValueError('Make sure setting the batch size to 1.')

    def _backward_pass(self, which_class):
        self.target_class = CamGenerator.class_dict[which_class]
        batch_size, one_hot_length = self.output.size()

        one_hot = torch.FloatTensor(1, one_hot_length).zero_()
        one_hot[0][self.target_class] = 1.0
        self.output.backward(one_hot, retain_graph = True)

    def find_gradient_CAM(self):
        self.gradient = self.gradient/(torch.sqrt(torch.mean(torch.pow(self.gradient, 2))) + 1e-5)
        batch_size, channels, size_of_gradient, _ = self.gradient.size()
        gradient_average = nn.AvgPool2d([size_of_gradient, size_of_gradient])(self.gradient) #get mean gradient for each layer
        assert self.length_of_filter == channels, "Number of filter does not match gradient dimension"
        gradient_average.resize_(channels) # lots of squeeze

        self.gradient_CAM = torch.FloatTensor(self.size_of_feature_maps, self.size_of_feature_maps).zero_() # 7 by 7 probably
        for feature_map, weight in zip(self.feature_maps, gradient_average):
                self.gradient_CAM = self.gradient_CAM + feature_map * weight.data

        self.gradient_CAM = CamGenerator._post_processing_for_cam(self.gradient_CAM)

    @staticmethod
    def _post_processing_for_cam(gradient_CAM):

        gradient_CAM = F.relu(gradient_CAM)
        gradient_CAM = gradient_CAM - gradient_CAM.min();
        gradient_CAM = gradient_CAM/gradient_CAM.max()
        gradient_CAM = cv2.resize(gradient_CAM.numpy(), (224, 224))
        gradient_CAM = cv2.applyColorMap(np.uint8(gradient_CAM * 255.0), cv2.COLORMAP_JET)

        return gradient_CAM

    def show(self):
        combined_image = self.gradient_CAM.astype(np.float) + self.image_numpy*255
        gradient_CAM = (combined_image/combined_image.max())*255

        try:
            # guessed_name = 'Positive' if self.target_class == 1 else 'Negative'
            # true_name = 'Positive' if self.label == 1 else 'Negative'
            fig = plt.figure()
            ax1 = fig.add_subplot(1,2,1)
            ax1.imshow(self.image_numpy)

           # plt.title('Ground Truth: {}'.format(true_name))
            ax2 = fig.add_subplot(1,2,2)
            ax2.imshow(gradient_CAM/255)
          #  plt.title('Prediction: {}'.format(guessed_name))
            plt.show()

        except AttributeError:
            print('Need to first call find_gradient_CAM()')
            return None

    def batch_testing_engine(self):
        pass
if __name__ == '__main__':
    model = torch.load('/Users/haigangliu/Desktop/ML_model_cache/multiclass/multi_class_.pth.tar', map_location = 'cpu')

    #print(cam_generator.feature_maps)

    transform_val = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    dataset_cam = SpecializedContructorForCAM("/Users/haigangliu/ImageData/ChestXrayData/", '/Users/haigangliu/ImageData/BBox_List_2017.csv', transform = transform_val)

    dataloader_cam = DataLoader(dataset = dataset_cam,batch_size = 1, shuffle = True, num_workers = 4, pin_memory = True)

    validation_dataset = ChestXrayDataSetMultiLabel('/Users/haigangliu/ImageData/ChestXrayData','/Users/haigangliu/ImageData/code/labels/test_list.txt', transform = transform_val)

    validation_dataloader = DataLoader(dataset = validation_dataset,batch_size = BATCH_SIZE,shuffle = True,num_workers = 4, pin_memory = True)
    cam_generator = CamGenerator(model,'features.denseblock4.denselayer16.conv2', validation_dataloader)
    cam_generator.load_data(validation_dataloader,'Effusion')
    cam_generator.find_gradient_CAM()
    cam_generator.show()
