import torch
import torch.nn as nn

class SCAR_LSTM(nn.Module):
    def __init__(self, orig_dim, hidden_size, num_layers, in_size=768):
        super(SCAR_LSTM, self).__init__()
        
        self.down_projector = nn.Linear(in_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.up_projector = nn.Linear(hidden_size, 768)

        self.orig_dim = orig_dim
        
    def forward(self, x):
        x = self.down_projector(x) #[b, T, hidden_size]
        x, _ = self.lstm(x) #[b, T, hidden_size]
        x = self.up_projector(x) #[b, T, 768]

        return x


class SCAR_TempX(nn.Module):
    def __init__(self, T=8):
        super(SCAR_TempX, self).__init__()
        self.pconv1d = nn.Sequential(
            *[
                nn.Conv1d(in_channels=197, out_channels=197, kernel_size=3, dilation=1, padding=1),
                nn.Conv1d(in_channels=197, out_channels=197, kernel_size=3, dilation=2, padding=2),
                nn.Conv1d(in_channels=197, out_channels=197, kernel_size=3, dilation=3, padding=3),
                # nn.Conv1d(in_channels=197, out_channels=197, kernel_size=3, dilation=8, padding=8),
            ]
        )

        self.plstm = SCAR_LSTM(768, 512, 2, 197)

        self.on_d = on_d
        self.on_p = on_p
        self.T = T

    def forward(self, x): # x: [197, b*T, 768]
        x = x.permute(1, 0, 2)  # [b*T, 197, 768]
        x = x.reshape(x.shape[0]//self.T, self.T, x.shape[1], x.shape[-1]) #[b, T, p, hidden_size]

        xp = torch.mean(x, dim=3) #[b, T, 197]
        xp = xp.permute(0, 2, 1) #[b, 197, T]
        xp = self.pconv1d(xp).permute(0, 2, 1) #[b, T, 197]
        x = self.plstm(xp) #[b, T, 768]

        x = x.reshape(x.shape[0]*x.shape[1], x.shape[-1]) #[b*T, 768]

        return x
        
# if __name__ == "__main__":
#     example_input = torch.randn(197, 4*8, 768)
#     model = SCAR_TempX()

#     output = model(example_input)
#     print("Output shape:", output.shape) 

# if __name__ == "__main__":
#     b, T = 32, 10
#     example_input = torch.randn(b, T, 196, 768)
#     model = SCAR_LSTM(768, 512, 2)

#     output = model(example_input)
#     print("Output shape:", output.shape) 