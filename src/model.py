from torch import nn
from sklearn.preprocessing import MinMaxScaler


class Scaler:
    """Normalize data. """
    def __init__(self, df):
        self.df = df
        self.minmaxscaler = MinMaxScaler().fit(df[['capacity_connected', 'charge_speed', 'energy_register']])

    def scaler(self, data):
        data_scaler = self.minmaxscaler.transform(data)
        return data_scaler

    def inverse(self, data):
        data_inverse = self.minmaxscaler.inverse_transform(data)
        return data_inverse


class Generator(nn.Module):
    """Generate adversarial data from noise. """
    def __init__(self, latent_dim, len_session):
        super().__init__()
        self.len_session = len_session
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256,),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, len_session * 2),
        )

        self.lstm = nn.LSTM(input_size=2, hidden_size=128, num_layers=2, batch_first=True)
        self.lstm1 = nn.LSTM(input_size=128, hidden_size=2, num_layers=2, batch_first=True)
        self.linear1 = nn.Linear(len_session * 2, len_session * 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        data = self.linear(z).view(z.size(0), self.len_session, 2)
        output, (hidden, cell) = self.lstm(data)
        output, (hidden, cell) = self.lstm1(output)
        output = self.linear1(output.reshape(output.size(0), self.len_session * 2)).reshape(output.size(0), self.len_session, 2)
        output = self.sigmoid(output)
        return output


class Discriminator(nn.Module):
    """Discriminate real data and fake data. """
    def __init__(self, len_session):
        super().__init__()
        self.len_session = len_session
        self.model = nn.Sequential(
            nn.Linear(len_session * 3, 1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        output = z.reshape(z.size(0), -1)
        output = self.model(output)
        return output
