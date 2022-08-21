from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet34
import torch
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TempAttnNet(nn.Module):
    def __init__(self, num_classes=2, drop_rate=0, attnvis = False):
        super(TempAttnNet, self).__init__()
        self.encoder = Encoder()
        self.low_maxpool = nn.MaxPool2d(8)

        # Temporal aggregation
        self.lstm = nn.LSTM(input_size=2048, hidden_size=256, batch_first=True)

        self.last_linear = nn.Linear(256, 1, bias=True)

    def attention_net(self, lstm_output, final_state):

        """ 
        Now we will incorporate Attention mechanism in our LSTM model. In this new model, we will use attention to compute soft alignment score corresponding
        between each of the hidden_state and the last hidden_state of the LSTM. We will be using torch.bmm for the batch matrix multiplication.
        
        Arguments
        ---------
        
        lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
        final_state : Final time-step hidden state (h_n) of the LSTM
        
        ---------
        
        Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
                  new hidden state.
                  
        Tensor Size :
                    hidden.size() = (batch_size, hidden_size)
                    attn_weights.size() = (batch_size, num_seq)
                    soft_attn_weights.size() = (batch_size, num_seq)
                    new_hidden_state.size() = (batch_size, hidden_size)
                      
        """
        
        hidden = final_state.squeeze(0)
        attn_weights = torch.bmm(lstm_output, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights, 1)
        new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        
        return new_hidden_state

    def forward(self, X):
        X = X[0,...].permute(1, 0, 2, 3) # framesx3xhxw
        X = self.encoder(X) # framesx2048x8x8
        X = X.unsqueeze(0)    # 1xframesx2048 (bs x seq_len x embed_dim)

        output, (final_hidden_state, final_cell_state) = self.lstm(X, None)
        # print(final_hidden_state.shape)
        # output = output.permute(1, 0, 2) # output.size() = (batch_size, num_seq, hidden_size)
        # print(output.shape)
        
        attn_output = self.attention_net(output, final_hidden_state)

        logits = self.last_linear(attn_output)

        logits = torch.sigmoid(logits)
        # logits = logits[:, -1, :]
        # print(logits.shape, logits)

        return logits

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.rgb_base = nn.Sequential(*list(resnet34(pretrained=True).children())[:-2])

        # Pooling
        self.low_avgpool = nn.AvgPool2d(8)

    def forward(self, x):
        x = self.rgb_base(x) # batchx512x8x8
        x = self.low_avgpool(x)     # batchx512x1x1
        x = x.squeeze()    # 1xframesx512 (bs x seq_len x embed_dim)
        return x


class Classifier(nn.Module):
    def __init__(self, n_classes, dropout=0.5):
        super(Classifier, self).__init__()
        self.l1 = nn.Linear(500, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.l1(x)
        return x


# class CNN(nn.Module):
#     def __init__(self, in_channels=1, n_classes=10, target=False):
#         super(CNN, self).__init__()
#         self.encoder = Encoder(in_channels=in_channels)
#         self.classifier = Classifier(n_classes)
#         if target:
#             for param in self.classifier.parameters():
#                 param.requires_grad = False

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.classifier(x)
#         return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.rgb_base = nn.Sequential(*list(resnet34(pretrained=True).children())[:-2])
        self.low_avgpool = nn.AvgPool2d(8)

    def forward(self, x):
        x = self.rgb_base(x) # batchx512x8x8
        x = self.low_avgpool(x)     # batchx512x1x1
        x = x.squeeze()    # 1xframesx512 (bs x seq_len x embed_dim)
        return x

class Discriminator(nn.Module):
    def __init__(self, h=500, args=None):
        super(Discriminator, self).__init__()
        self.l1 = nn.Linear(512, h)
        self.l2 = nn.Linear(h, h)
        self.l3 = nn.Linear(h, 2)
        self.slope = args.slope

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = F.leaky_relu(self.l1(x), self.slope)
        x = F.leaky_relu(self.l2(x), self.slope)
        x = self.l3(x)
        return x
