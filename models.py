import torch
import torch.nn as nn

class C3D(nn.Module):
    """
    The C3D network model.
    """
    def __init__(self):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):

        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        h = h.view(-1, 8192) # Flatten the features
        h = self.relu(self.fc6(h))

        return h

class LSTM(nn.Module):
    def __init__(self, input_size=4096, hidden_size=256, num_layers=1, batch_first=True):
        super(LSTM, self).__init__()
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first
        )
        # Regression
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: [(batch_size, seq_length, feature_dim)]
        lstm_out, (h_n, c_n) = self.lstm(x)

        score = self.fc(lstm_out[:, -1, :]) # last time step

        return score.squeeze(1)

class CompleteModel(nn.Module):
    def __init__(self):
        super(CompleteModel, self).__init__()
        self.c3d = C3D()
        self.lstm = LSTM()

    def forward(self, x):
        # x: (B, T, C, H, W)

        clips = []
        for i in range(0, x.size(1) - 15, 16): # Clips of 16 frames
            clip = x[:, i:i+16, :, :, :]
            clips.append(clip)

        # Batch of clips
        clips = torch.stack(clips)
        batch_size = clips.size(1)
        clips = clips.view(-1, clips.size(2), clips.size(3), clips.size(4), clips.size(5))

        # C3D: (B, C, T, H, W)
        clips = clips.permute(0, 2, 1, 3, 4)

        # Models
        clip_features = self.c3d(clips)
        clip_features = clip_features.view(batch_size, -1, 4096)
        score = self.lstm(clip_features)

        return score

    def load_pretrained_weights(self, c3d_pretrained_path):

        pretrained_dict = torch.load(c3d_pretrained_path)
        c3d_dict = self.c3d.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in c3d_dict}

        c3d_dict.update(pretrained_dict)
        self.c3d.load_state_dict(c3d_dict)
        print("C3D weights loaded")
