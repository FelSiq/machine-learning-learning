import typing as t

from . import base
from . import _utils


class Sequential(base.BaseLayer):
    def __init__(self, layers):
        assert len(layers)

        super(Sequential, self).__init__(trainable=True)

        layers = _utils.collapse(
            layers, atom=base.BaseComponent, exceptions=(Sequential)
        )
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

    def __repr__(self):
        strs = [f"Sequential component with {len(self)} layers:"]

        if self.frozen:
            strs[0] += " (frozen)"

        for i, layer in enumerate(self.layers):
            strs.append(f" | {i}. {str(layer)}")

        return "\n".join(strs)


class SkipConnection(base.BaseLayer):
    def __init__(
        self,
        layer_main: base.BaseComponent,
        layer_skip: t.Optional[base.BaseComponent] = None,
        layer_combine: t.Optional[base.BaseComponent] = None,
        activation: t.Optional[base.BaseComponent] = None,
    ):
        super(SkipConnection, self).__init__(trainable=True)

        self.layer_main = layer_main

        self.layer_combine = None
        self.layer_skip = None
        self.activation = None

        self.register_layers(self.layer_main)

        if layer_skip is not None:
            self.layer_skip = layer_skip
            self.register_layers(self.layer_skip)

        if layer_combine is not None:
            self.layer_combine = layer_combine

        else:
            self.layer_combine = base.Add()

        self.register_layers(self.layer_combine)

        if activation is not None:
            self.activation = activation
            self.register_layers(self.activation)

    def forward(self, X):
        X_skip = X
        X_main = self.layer_main(X)

        if self.layer_skip is not None:
            X_skip = self.layer_skip(X_skip)

        out = self.layer_combine(X_main, X_skip)

        if self.activation is not None:
            out = self.activation(out)

        return out

    def backward(self, dout):
        if self.activation is not None:
            dout = self.activation.backward(dout)

        dX_main, dX_skip = self.layer_combine.backward(dout)

        dX = self.layer_main.backward(dX_main)

        if self.layer_skip is not None:
            dX_skip = self.layer_skip.backward(dX_skip)

        dX += dX_skip

        return dX
