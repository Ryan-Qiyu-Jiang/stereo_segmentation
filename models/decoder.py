from stero_segmentation.monodepth2.layers import *
from collections import OrderedDict
import sys, os
sys.path.append(os.path.abspath("../monodepth2"))
import networks

# import stero_segmentation.monodepth2.networks as networks


class DebugDepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DebugDepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x1 = self.convs[("upconv", i, 0)](x)
            x2 = [upsample(x1)]
            if self.use_skips and i > 0:
                x2 += [input_features[i - 1]]
            x3 = torch.cat(x2, 1)
            x4 = self.convs[("upconv", i, 1)](x3)

            self.outputs[("debug_0", i)] = x
            self.outputs[("debug_1", i)] = x1
            self.outputs[("debug_2", i)] = x3
            self.outputs[("debug", i)] = x4
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x4))
            x = x4

        return self.outputs

# class CAMSeedsModel():
#   def __init__(self):
#     self.net = models.resnet18(pretrained=True)
#     self.net.eval()
#     params = list(net.parameters())
#     self.weight_softmax = np.squeeze(params[-2].data.numpy())
#     self.feature_blobs = []
#     # LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
#     # self.classes = {int(key):value for (key, value)
#     #       in requests.get(LABELS_URL).json().items()}
#     def hook_l4(module, input, output):
#       self.feature_blobs.append(output.data.cpu().numpy())
#     self.net._modules.get('layer4').register_forward_hook(hook_l4)

#   def get_CAM(self, feature_conv, weight_softmax, class_idx):
#       # generate the class activation maps upsample to 256x256
#       size_upsample = (256, 256)
#       bz, nc, h, w = feature_conv.shape
#       output_cam = []
#       for idx in class_idx:
#           cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
#           cam = cam.reshape(h, w)
#           cam = cam - np.min(cam)
#           cam_img = cam / np.max(cam)
#           cam_img = np.uint8(255 * cam_img)
#           output_cam.append(cv2.resize(cam_img, size_upsample))
#       return output_cam

#   def get_seeds(self, x):
#     logit = self.net(x)
#     idx = np.argmax(logit)
#     cam = self.get_CAM(self.features_blobs[-1], self.weight_softmax, [idx])
#     return cam