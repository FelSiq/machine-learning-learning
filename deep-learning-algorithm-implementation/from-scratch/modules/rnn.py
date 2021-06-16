import numpy as np

import base


class RNNCell(base._BaseLayer):
    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int):
        super(RNNCell, self).__init__()

        assert dim_in > 0
        assert dim_hidden > 0
        assert dim_out > 0

        self.lin_hidden = base.Linear(dim_hidden, dim_hidden, include_bias=False)
        self.lin_input = base.Linear(dim_in, dim_hidden, include_bias=True)

        self.cell_state = np.zeros(dim_hidden, dtype=float)

    def forward(self, X):
        aux_cell_state = self.lin_hidden(self.cell_state)
        aux_X = self.lin_input(X)

        self.cell_state = np.tanh(aux_cell_state + aux_X)

        self._store_in_cache(self.cell_state)

        return self.cell_state

    def backward(self, dout_h, dout_y):
        (cell_state,) = self._pop_from_cache()

        dcs = (1.0 - np.square(cell_state)) * (dout_h + dout_y)

        (dh, dWh, dbh) = self.lin_hidden.backward(dcs)
        (dX, dWX, dbX) = self.lin_input.backward(dcs)

        return (dh, dX, dWh, dbh, dWX, dbX)

    def update(self, *args):
        if self.frozen:
            return

        (dWh, dbh, dWX, dbX) = args

        self.lin_hidden.update(dWh, dbh)
        self.lin_input.update(dWX, dbX)
