import torch
import torch.nn as nn

class SCAR_LSTM(nn.Module):
    def __init__(self, orig_dim, hidden_size, num_layers):
        super(SCAR_LSTM, self).__init__()
        
        self.down_projector = nn.Linear(768, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.up_projector = nn.Linear(hidden_size*2, 768)

        self.orig_dim = orig_dim
        
    def forward(self, x):
        x = self.down_projector(x)
        x = torch.mean(x, dim=2)
        x, _ = self.lstm(x)
        x = self.up_projector(x)

        return x

if __name__ == "__main__":
    b, T = 32, 10
    example_input = torch.randn(b, T, 196, 768)
    model = SCAR_LSTM(768, 512, 2)

    output = model(example_input)
    print("Output shape:", output.shape) 