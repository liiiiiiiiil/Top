import numpy as np
import torch.nn as nn
from .model.resnet import *
import torch



class Resnet101_8s(nn.Module):
    
    
    def __init__(self, num_classes=1000):
        
        super(Resnet101_8s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet101_8s = resnet101(pretrained=True)
        
        self.last_conv = nn.Conv2d(2048,num_classes,kernel_size=1)
        
        self.resnet101_8s = resnet101_8s
        
        self._normal_initialization(self.resnet101_8s.fc)
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        x = self.resnet101_8s(x)
        x = self.last_conv(x)
    
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        
        return x
class Resnet101_16s(nn.Module):
    
    
    def __init__(self, num_classes=1000):
        
        super(Resnet101_16s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet101_16s = resnet101(fully_conv=True,
                                        pretrained=True,
                                        output_stride=16,
                                        remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet101_16s.fc = nn.Conv2d(resnet101_16s.inplanes, num_classes, 1)
        
        self.resnet101_16s = resnet101_16s
        
        self._normal_initialization(self.resnet101_16s.fc)
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet101_16s(x)
        
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        
        return x   

    
class Resnet18_8s(nn.Module):
    
    
    def __init__(self, num_classes=1000):
        
        super(Resnet18_8s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet18_8s = resnet18(fully_conv=True,
                                      pretrained=True,
                                      output_stride=8,
                                      remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet18_8s.fc = nn.Conv2d(resnet18_8s.inplanes, num_classes, 1)
        
        self.resnet18_8s = resnet18_8s
        
        self._normal_initialization(self.resnet18_8s.fc)
                
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x, feature_alignment=False):
        
        input_spatial_dim = x.size()[2:]
        
        if feature_alignment:
            
            x = adjust_input_image_size_for_proper_feature_alignment(x, output_stride=8)
        
        x = self.resnet18_8s(x)
        
        x = nn.functional.upsample(x, size=input_spatial_dim, mode='bilinear', align_corners=True)
        
        #x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)#, align_corners=False)
        
        return x

    
class Resnet18_16s(nn.Module):
    
    
    def __init__(self, num_classes=1000):
        
        super(Resnet18_16s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 16
        resnet18_16s = resnet18(fully_conv=True,
                                      pretrained=True,
                                      output_stride=16,
                                      remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet18_16s.fc = nn.Conv2d(resnet18_16s.inplanes, num_classes, 1)
        
        self.resnet18_16s = resnet18_16s
        
        self._normal_initialization(self.resnet18_16s.fc)
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet18_16s(x)
        
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        
        return x
    

class Resnet18_32s(nn.Module):
    
    
    def __init__(self, num_classes=1000):
        
        super(Resnet18_32s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet18_32s = resnet18(fully_conv=True,
                                       pretrained=True,
                                       output_stride=32,
                                       remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet18_32s.fc = nn.Conv2d(resnet18_32s.inplanes, num_classes, 1)
        
        self.resnet18_32s = resnet18_32s
        
        self._normal_initialization(self.resnet18_32s.fc)
        import numpy as np
import torch.nn as nn
import torchvision.models as models
import torch
import math


class PSP_head(nn.Module):
    
    def __init__(self, in_channels):
        
        super(PSP_head, self).__init__()
        
        out_channels = int( in_channels / 4 )
        
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))
        
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))
        
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))
        
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                   nn.BatchNorm2d(out_channels),
                                   nn.ReLU(True))
        
        self.fusion_bottleneck = nn.Sequential(nn.Conv2d(in_channels * 2, out_channels, 3, padding=1, bias=False),
                                               nn.BatchNorm2d(out_channels),
                                               nn.ReLU(True),
                                               nn.Dropout2d(0.1, False))
        
        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        
    def forward(self, x):

        fcn_features_spatial_dim = x.size()[2:]

        pooled_1 = nn.functional.adaptive_avg_pool2d(x, 1)
        pooled_1 = self.conv1(pooled_1)
        pooled_1 = nn.functional.upsample_bilinear(pooled_1, size=fcn_features_spatial_dim)

        pooled_2 = nn.functional.adaptive_avg_pool2d(x, 2)
        pooled_2 = self.conv2(pooled_2)
        pooled_2 = nn.functional.upsample_bilinear(pooled_2, size=fcn_features_spatial_dim)

        pooled_3 = nn.functional.adaptive_avg_pool2d(x, 3)
        pooled_3 = self.conv3(pooled_3)
        pooled_3 = nn.functional.upsample_bilinear(pooled_3, size=fcn_features_spatial_dim)

        pooled_4 = nn.functional.adaptive_avg_pool2d(x, 6)
        pooled_4 = self.conv4(pooled_4)
        pooled_4 = nn.functional.upsample_bilinear(pooled_4, size=fcn_features_spatial_dim)

        x = torch.cat([x, pooled_1, pooled_2, pooled_3, pooled_4],
                       dim=1)

        x = self.fusion_bottleneck(x)

        return x

        
class Resnet50_8s_psp(nn.Module):
    
    def __init__(self, num_classes=1000):
        
        super(Resnet50_8s_psp, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 16
        resnet50_8s = resnet50(fully_conv=True,
                                      pretrained=True,
                                      output_stride=8,
                                      remove_avg_pool_layer=True)
        
        self.psp_head = PSP_head(resnet50_8s.inplanes)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet50_8s.fc = nn.Conv2d(resnet50_8s.inplanes // 4, num_classes, 1)
        
        self.resnet50_8s = resnet50_8s
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet50_8s.conv1(x)
        x = self.resnet50_8s.bn1(x)
        x = self.resnet50_8s.relu(x)
        x = self.resnet50_8s.maxpool(x)

        x = self.resnet50_8s.layer1(x)
        x = self.resnet50_8s.layer2(x)
        x = self.resnet50_8s.layer3(x)
        x = self.resnet50_8s.layer4(x)
        
        x = self.psp_head(x)
        
        x = self.resnet50_8s.fc(x)
        
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        
        return x
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet18_32s(x)
        
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        
        return x
    

    
class Resnet34_32s(nn.Module):
    
    
    def __init__(self, num_classes=1000):
        
        super(Resnet34_32s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet34_32s = resnet34(fully_conv=True,
                                       pretrained=True,
                                       output_stride=32,
                                       remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet34_32s.fc = nn.Conv2d(resnet34_32s.inplanes, num_classes, 1)
        
        self.resnet34_32s = resnet34_32s
        
        self._normal_initialization(self.resnet34_32s.fc)
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet34_32s(x)
        
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        
        return x

    
class Resnet34_16s(nn.Module):
    
    
    def __init__(self, num_classes=1000):
        
        super(Resnet34_16s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet34_16s = resnet34(fully_conv=True,
                                       pretrained=True,
                                       output_stride=16,
                                       remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet34_16s.fc = nn.Conv2d(resnet34_16s.inplanes, num_classes, 1)
        
        self.resnet34_16s = resnet34_16s
        
        self._normal_initialization(self.resnet34_16s.fc)
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet34_16s(x)
        
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        
        return x


class Resnet34_8s(nn.Module):
    
    
    def __init__(self, num_classes=1000):
        
        super(Resnet34_8s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet34_8s = resnet34(fully_conv=True,
                                       pretrained=True,
                                       output_stride=8,
                                       remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet34_8s.fc = nn.Conv2d(resnet34_8s.inplanes, num_classes, 1)
        
        self.resnet34_8s = resnet34_8s
        
        self._normal_initialization(self.resnet34_8s.fc)
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x, feature_alignment=False):
        
        input_spatial_dim = x.size()[2:]
        
        if feature_alignment:
            
            x = adjust_input_image_size_for_proper_feature_alignment(x, output_stride=8)
        
        x = self.resnet34_8s(x)
        
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        
        return x
    
class Resnet50_32s(nn.Module):
    
    
    def __init__(self, num_classes=1000):
        
        super(Resnet50_32s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet50_32s = resnet50(fully_conv=True,
                                       pretrained=True,
                                       output_stride=32,
                                       remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet50_32s.fc = nn.Conv2d(resnet50_32s.inplanes, num_classes, 1)
        
        self.resnet50_32s = resnet50_32s
        
        self._normal_initialization(self.resnet50_32s.fc)
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet50_32s(x)
        
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        
        return x

    
class Resnet50_16s(nn.Module):
    
    
    def __init__(self, num_classes=1000):
        
        super(Resnet50_16s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 16
        resnet50_8s = resnet50(fully_conv=True,
                                       pretrained=True,
                                       output_stride=16,
                                       remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet50_8s.fc = nn.Conv2d(resnet50_8s.inplanes, num_classes, 1)
        
        self.resnet50_8s = resnet50_8s
        
        self._normal_initialization(self.resnet50_8s.fc)
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet50_8s(x)
        
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        
        return x

class Resnet50_8s(nn.Module):
    
    
    def __init__(self, num_classes=1000):
        
        super(Resnet50_8s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet50_8s = resnet50(fully_conv=True,
                                       pretrained=True,
                                       output_stride=8,
                                       remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet50_8s.fc = nn.Conv2d(resnet50_8s.inplanes, num_classes, 1)
        
        self.resnet50_8s = resnet50_8s
        
        self._normal_initialization(self.resnet50_8s.fc)
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet50_8s(x)
        
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        
        return x

    

class Resnet9_8s(nn.Module):
    
    # Gets ~ 46 MIOU on Pascal Voc
    
    def __init__(self, num_classes=1000):
        
        super(Resnet9_8s, self).__init__()
        
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet18_8s = resnet18(fully_conv=True,
                                      pretrained=True,
                                      output_stride=8,
                                      remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet18_8s.fc = nn.Conv2d(resnet18_8s.inplanes, num_classes, 1)
        
        self.resnet18_8s = resnet18_8s
        
        self._normal_initialization(self.resnet18_8s.fc)
        
        
    def _normal_initialization(self, layer):
        
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        
        input_spatial_dim = x.size()[2:]
        
        x = self.resnet18_8s.conv1(x)
        x = self.resnet18_8s.bn1(x)
        x = self.resnet18_8s.relu(x)
        x = self.resnet18_8s.maxpool(x)

        x = self.resnet18_8s.layer1[0](x)
        x = self.resnet18_8s.layer2[0](x)
        x = self.resnet18_8s.layer3[0](x)
        x = self.resnet18_8s.layer4[0](x)
        
        x = self.resnet18_8s.fc(x)
        
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        
        return x


if __name__ == '__main__':
    model = Resnet9_8s()
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size)