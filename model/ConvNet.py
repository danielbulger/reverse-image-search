from torch import nn
import torchvision.models as models
import torch.nn.functional as F


class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self._setup_model()
        self._freeze()

    def _setup_model(self):
        # Use the VGG16 model without the two fully connected layers
        self.model = nn.Sequential(*list(
            models.vgg16(pretrained=True).children()
        )[:-1])

        self.fc1 = nn.Linear(in_features=512 * 7 * 7, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.dropout = nn.Dropout(p=0.6)
        self.relu = nn.ReLU()

    def _freeze(self):
        # We don't want to tweak the pre-trained model so we
        # freeze all the parameters.
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.model(x)

        # Flatten the output of the CNN
        x = x.view(x.size()[0], -1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.relu(self.fc2(x))
        x = self.dropout(x)

        # L2 normalize the output
        return F.normalize(x, p=2, dim=1)
