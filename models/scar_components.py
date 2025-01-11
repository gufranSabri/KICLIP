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
        self.conv1d_p = nn.Sequential(
            *[
                nn.Conv1d(in_channels=197, out_channels=197, kernel_size=3, dilation=1, padding=1),
                nn.Conv1d(in_channels=197, out_channels=197, kernel_size=3, dilation=2, padding=2),
                nn.Conv1d(in_channels=197, out_channels=197, kernel_size=3, dilation=3, padding=3),
            ]
        )

        # self.conv1d_d = nn.Sequential(
        #     *[
        #         nn.Conv1d(in_channels=768, out_channels=768, kernel_size=3, dilation=1, padding=1),
        #         nn.Conv1d(in_channels=768, out_channels=768, kernel_size=3, dilation=2, padding=2),
        #         nn.Conv1d(in_channels=768, out_channels=768, kernel_size=3, dilation=3, padding=3),
        #     ]
        # )

        self.lstm_p = SCAR_LSTM(768, 512, 2, 197)
        # self.lstm_d = SCAR_LSTM(768, 512, 2, 768)

        # self.conv_pooler = nn.Conv1d(in_channels=768, out_channels=768, kernel_size=2, stride=2)

        self.T = T

    def forward(self, x): # x: [197, b*T, 768]
        x = x.permute(1, 0, 2)  # [b*T, 197, 768]
        x = x.reshape(x.shape[0]//self.T, self.T, x.shape[1], x.shape[-1]) #[b, T, p, hidden_size]

        xp = torch.mean(x, dim=3) #[b, T, 197]
        xp = xp.permute(0, 2, 1) #[b, 197, T]
        xp = self.conv1d_p(xp).permute(0, 2, 1) #[b, T, 197]
        xp = self.lstm_p(xp) #[b, T, 768]
        x = xp

        # xd = torch.mean(x, dim=2) #[b, T, 768]
        # xd = xd.permute(0, 2, 1) #[b, 768, T]
        # xd = self.conv1d_d(xd).permute(0, 2, 1) #[b, T, 768]
        # xd = self.lstm_d(xd) #[b, T, 768]
        # x = xd

        # x = torch.cat((xp, xd), dim=1).permute(0, 2, 1) #[b, 768, T*2]
        # x = self.conv_pooler(x).permute(0, 2, 1) #[b, T, 768]
        x = x.reshape(x.shape[0]*x.shape[1], x.shape[-1]) #[b*T, 768]

        return x

class DualCrossAttention(nn.Module):
    def __init__(self, d_model=768, n_heads=8):
        super(DualCrossAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.mha1 = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.mha2 = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

        self.projection = nn.Linear(d_model * 2, d_model)

    def forward(self, x1, x2):
        out1, _ = self.mha1(query=x2, key=x1, value=x1)  # [b, T, d_model]
        out2, _ = self.mha2(query=x1, key=x2, value=x2)  # [b, T, d_model]

        combined = torch.cat([out1, out2], dim=-1)  # [b, T, 2 * d_model]
        output = self.projection(combined)          # [b, T, d_model]

        return output

        
if __name__ == "__main__":
    # n_heads = 8
    # b, T, d_model = 4, 8, 768
    # x1 = torch.randn(b, T, d_model)
    # x2 = torch.randn(b, T, d_model)

    # dual_cross_attention = DualCrossAttention(d_model=d_model, n_heads=n_heads)
    # output = dual_cross_attention(x1, x2)
    # print("Output shape:", output.shape)  # Should be [b, T, d_model]


    example_input = torch.randn(197, 4*8, 768)
    model = SCAR_TempX()

    output = model(example_input)
    print("Output shape:", output.shape) 


    # b, T = 32, 10
    # example_input = torch.randn(b, T, 196, 768)
    # model = SCAR_LSTM(768, 512, 2)

    # output = model(example_input)
    # print("Output shape:", output.shape) 