import torch


class MinMaxScaler:
    """
    Min-max scaler that is fit on training data and reused for validation/test data.
    """
    def __init__(self, feature_dims=None, out_range=(0., 1.), eps=1e-12, clamp=True):
        self.feature_dims = feature_dims
        self.out_range = out_range
        self.eps = eps
        self.clamp = clamp
        self.data_min = None
        self.data_max = None

    def fit(self, x):
        reduce_dims = self._get_reduce_dims(x)
        self.data_min = x.amin(dim=reduce_dims, keepdim=True)
        self.data_max = x.amax(dim=reduce_dims, keepdim=True)
        return self

    def transform(self, x):
        assert self.data_min is not None and self.data_max is not None, 'MinMaxScaler must be fit before transform.'
        data_min = self.data_min.to(device=x.device, dtype=x.dtype)
        data_max = self.data_max.to(device=x.device, dtype=x.dtype)
        x = (x - data_min) / (data_max - data_min).clamp_min(self.eps)

        out_min, out_max = self.out_range
        x = x * (out_max - out_min) + out_min
        if self.clamp:
            x = x.clamp(min=out_min, max=out_max)
        return x

    def fit_transform(self, x):
        return self.fit(x).transform(x)

    def _get_reduce_dims(self, x):
        if self.feature_dims is None:
            return tuple(range(x.ndim))

        feature_dims = self.feature_dims
        if isinstance(feature_dims, int):
            feature_dims = (feature_dims,)
        feature_dims = tuple(dim if dim >= 0 else x.ndim + dim for dim in feature_dims)
        return tuple(dim for dim in range(x.ndim) if dim not in feature_dims)


class StandardScaler:
    """
    Standard scaler that is fit on training data and reused for validation/test data.
    """
    def __init__(self, feature_dims=None, eps=1e-12):
        self.feature_dims = feature_dims
        self.eps = eps
        self.mean = None
        self.std = None

    def fit(self, x):
        reduce_dims = self._get_reduce_dims(x)
        self.mean = x.mean(dim=reduce_dims, keepdim=True)
        self.std = x.std(dim=reduce_dims, keepdim=True, unbiased=False)
        return self

    def transform(self, x):
        assert self.mean is not None and self.std is not None, 'StandardScaler must be fit before transform.'
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std.clamp_min(self.eps)

    def fit_transform(self, x):
        return self.fit(x).transform(x)

    def _get_reduce_dims(self, x):
        if self.feature_dims is None:
            return tuple(range(x.ndim))

        feature_dims = self.feature_dims
        if isinstance(feature_dims, int):
            feature_dims = (feature_dims,)
        feature_dims = tuple(dim if dim >= 0 else x.ndim + dim for dim in feature_dims)
        return tuple(dim for dim in range(x.ndim) if dim not in feature_dims)


class ThermometerEncoding(torch.nn.Module):
    """
    Convert continuous inputs into threshold channels.
    """
    def __init__(
            self,
            num_bits: int,
            value_range=(0., 1.),
            thresholds=None,
            dtype=torch.float32,
            flatten_channels=True,
    ):
        super().__init__()
        self.num_bits = num_bits
        self.value_range = value_range
        self.dtype = dtype
        self.flatten_channels = flatten_channels

        if thresholds is None:
            low, high = value_range
            thresholds = torch.linspace(low, high, num_bits + 2, dtype=torch.float32)[1:-1]
        else:
            thresholds = torch.as_tensor(thresholds, dtype=torch.float32)
            assert thresholds.ndim == 1, thresholds.shape
            assert thresholds.shape[0] == num_bits, (thresholds.shape, num_bits)

        self.register_buffer('thresholds', thresholds)

    def forward(self, x):
        thresholds = self._reshape_thresholds(x)
        encoded = (x.unsqueeze(2) > thresholds).to(self.dtype)

        if self.flatten_channels:
            batch_size, channels, bits = encoded.shape[:3]
            return encoded.reshape(batch_size, channels * bits, *encoded.shape[3:])
        return encoded

    def _reshape_thresholds(self, x):
        shape = [1, 1, self.num_bits] + [1] * (x.ndim - 2)
        return self.thresholds.to(device=x.device, dtype=x.dtype).reshape(shape)

    def extra_repr(self):
        return 'num_bits={}, value_range={}, flatten_channels={}'.format(
            self.num_bits,
            self.value_range,
            self.flatten_channels,
        )
