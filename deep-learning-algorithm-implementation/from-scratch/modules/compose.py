from . import base
from . import _utils


class Sequential(base.BaseLayer):
    def __init__(self, layers):
        super(Sequential, self).__init__(trainable=True)
        layers = _utils.collapse(layers, base.BaseComponent)
        self.register_layers(*layers)

    def forward(self, X):
        out = X

        for layer in self.layers:
            out = layer(out)

        return out

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

        return dout

    def __getitem__(self, i):
        return self.layers[i]

    def __len__(self):
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)
