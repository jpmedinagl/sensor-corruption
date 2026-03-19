from data import ACCL, GYRO
import data
import numpy as np


rng = np.random.default_rng(20)


class CorruptionFramework:
    """
    corruption_type: str
        'stochastic'    - severity = fraction of std (0.25, 0.5, 1.0, 1.25)
        'dropout'       - severity = fraction of timesteps (0.0, ..., 1.0)
        'bias'          - severity = fraction of std (0.5, 1.0, 1.25, 2.0)
        'gain'          - severity = multiplier (0.5, 0.75, 1.25, 2.0)
        'drift'         - severity = multiplier of std (1, 2, 3, 4)
        'resolution'    - severity = bits retained (1, 2, 3, 4, 5, 6)
    """
    

    def __init__(self, corruption_type, channels, severity):
        self.corruption_type = corruption_type
        self.channels = channels
        self.severity = severity


    def _check_corruption(self):

        if (self.corruption_type == "stochastic"):
            if not 0 <= self.severity:
                raise ValueError("Severity must be positive for stochastic corruption")
        elif (self.corruption_type == "dropout"):
            if not 0 <= self.severity <= 1.0:
                raise ValueError("Severity must be between 0.0 and 1.0 for dropout corruption")
        elif (self.corruption_type == "bias"):
            if not 0 <= self.severity:
                raise ValueError("Severity must be positive for bias corruption")
        elif (self.corruption_type == "gain"):
            if not 0 <= self.severity:
                raise ValueError("Severity must be positive for gain corruption")
        elif (self.corruption_type == "drift"):
            if not 0 <= self.severity:
                raise ValueError("Severity must be positive for drift corruption")
        elif (self.corruption_type == "resolution"):
            if not 1 <= self.severity <= 6:
                raise ValueError("Severity must be between 1 and 6 for resolution corruption")
        else:
            raise ValueError("Corruption type must be one of: gaussian, dropout, drift, resolution")

    
    def _stochastic_corruption(self, X):
        X_c = X.copy()

        channel_std = X_c[:,:,self.channels].std(axis=1, keepdims=True)
        noise = rng.normal(0, 1, X_c[:, :, self.channels].shape)
        X_c[:, :, self.channels] += self.severity * channel_std * noise

        return X_c


    def _dropout_corruption(self, X):
        X_c = X.copy()
        samples = X.shape[0]
        timesteps = X.shape[1]

        n_corrupt = int(self.severity * timesteps)

        for i in range(samples):
            idx = rng.choice(timesteps, n_corrupt, replace=False)
            # X_c[i, idx, self.channels] = 0
            X_c[i][np.ix_(idx, self.channels)] = 0

        return X_c


    def _bias_corruption(self, X):
        X_c = X.copy()

        channel_std = X_c[:,:,self.channels].std(axis=1, keepdims=True)
        X_c[:,:,self.channels] += self.severity * channel_std

        return X_c


    def _gain_corruption(self, X):
        X_c = X.copy()
        X_c[:,:,self.channels] = X_c[:,:,self.channels] * self.severity
        return X_c


    def _drift_corruption(self, X):
        X_c = X.copy()
        timesteps = X.shape[1]
        
        channel_std = X_c[:,:,self.channels].std(axis=1, keepdims=True)
        drift = np.linspace(0, self.severity, timesteps)[None,:,None]

        X_c[:,:,self.channels] += drift * channel_std
        
        return X_c


    def _resolution_corruption(self, X):
        X_c = X.copy()
        X_c[:,:,self.channels] = np.round(X_c[:,:,self.channels], self.severity)
        return X_c


    def _apply_corruption(self, X):
        if (self.corruption_type == "stochastic"):
            return self._stochastic_corruption(X)
        elif (self.corruption_type == "dropout"):
            return self._dropout_corruption(X)
        elif (self.corruption_type == "bias"):
            return self._bias_corruption(X)
        elif (self.corruption_type == "gain"):
            return self._gain_corruption(X)
        elif (self.corruption_type == "drift"):
            return self._drift_corruption(X)
        elif (self.corruption_type == "resolution"):
            return self._resolution_corruption(X)
        else:
            return X


    def corrupt(self, X):
        self._check_corruption()
        return self._apply_corruption(X)


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = data.load_raw_data()

    framework = CorruptionFramework("dropout", channels=[GYRO[0]], severity=0.5)
    
    # data is corrupted
    X_train_corrupted = framework.corrupt(X_train)

    print(X_train[0,:,GYRO[0]])
    print(X_train_corrupted[0,:,GYRO[0]])
