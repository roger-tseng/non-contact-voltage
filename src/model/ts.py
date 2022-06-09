import torch
import torch.nn as nn

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

class ConvNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ConvNet, self).__init__()

        # Define your neural network here
        # TODO: How to modify this model to achieve better performance?
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=400, kernel_size=5, stride=3, padding=2, padding_mode='reflect'),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Conv1d(in_channels=400, out_channels=400, kernel_size=5, stride=3, padding=2, padding_mode='reflect'),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Conv1d(in_channels=400, out_channels=400, kernel_size=3, stride=2, padding=1, padding_mode='reflect'),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Conv1d(in_channels=400, out_channels=400, kernel_size=3, stride=2, padding=1, padding_mode='reflect'),
            nn.BatchNorm1d(400),
            nn.ReLU(),
            nn.Conv1d(in_channels=400, out_channels=64, kernel_size=3, stride=2, padding=1, padding_mode='reflect'),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        # attributes for transformer encoder layer
        d_model = 64
        dropout = 0.1
        dim_feedforward = 256
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=2, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.dropout_ff1 = nn.Dropout(dropout)
        self.dropout_ff2 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        #self.ts = self.ts_encoder
        self.ts = nn.TransformerEncoderLayer(
            d_model=64, dim_feedforward=256, nhead=2
        )
        self.fc = nn.Sequential(
            nn.Linear(768, 800),
            #nn.ReLU(),
            #nn.Linear(output_dim, output_dim),
        )

        #self.apply(init_weights)
    def _sa_block(self, x):
        x = self.attn(x, x, x,
            need_weights=False)[0]
        x = self.norm(x + self.dropout(x))
        return x
    
    def _ff_block(self, x):
        return self.norm2(x + self.dropout_ff2(self.linear2(self.dropout_ff1(self.activation(self.linear1(x))))))

    def ts_encoder(self, x):
        return self._ff_block(self._sa_block(x))

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.net(x)
        #print(f"after net: {x.shape}")
        #x = x.flatten(1,2)
        #print(f"after flat: {x.shape}")
        x = torch.transpose(x, 1, 2)
        #print(f"after transpose {x.shape}")
        x = self.ts(x)
        #print(f"after ts {x.shape}")
        x = x.flatten(1,2)
        x = self.fc(x)
        #print(f"after fc {x.shape}")
        return x#.squeeze(dim=-1)

class TS(nn.Module):
    def __init__(self, roll_dim, ff_dim, n_head):
        super(TS, self).__init__()
        self.roll_dim = roll_dim
        self.pad = nn.ReflectionPad1d((0,roll_dim-1))
        self.ts = nn.Sequential(
            nn.TransformerEncoderLayer(
                d_model=roll_dim, dim_feedforward=ff_dim, nhead=n_head, batch_first=True
            ),
        )
        self.fc = nn.Sequential(
            nn.Linear(roll_dim, 1),
            #nn.ReLU(),
            #nn.Linear(output_dim, output_dim),
        )

    def forward(self, x):
        x = torch.cat((x, x[:,:self.roll_dim-1]), dim=1).unfold(1, self.roll_dim, 1)
        #print(x.shape)
        x = self.ts(x)
        #print(f"after ts {x.shape}")
        x = self.fc(x)
        #print(f"after fc {x.shape}")
        return x.squeeze(dim=-1)

class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, hidden_dim2=None):
        super(NeuralNet, self).__init__()
        if not hidden_dim2:
            hidden_dim2 = output_dim
        # Define your neural network here
        # TODO: How to modify this model to achieve better performance?
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, output_dim),
            # nn.ReLU(),
            # nn.Linear(output_dim, output_dim),
        )

        #self.apply(init_weights)

    def forward(self, x):
        return self.net(x)

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
            
        # One time step
        out, hn = self.rnn(x, h0)
        out = self.fc(out[:, -1, :]) 
        return out