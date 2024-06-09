import gurobipy as gp
from gurobipy import GRB


class VModel:
    """
    The VModel class is the model class of VerNN.
    """
    def __init__(self, layers):
        self.layers = layers
        self.num_layers = len(layers)
        self.time_step = 0

    def forward(self):
        """
        The forward pass of the model.
        """

        # Forward pass through the layers
        for i in range(self.num_layers):
            if i:
                self.layers[i].forward(self.layers[i-1])
            else:
                self.layers[i].forward()

        # Update the time step
        self.time_step += 1


class Layer:
    def __init__(self, grb_model, idx):
        self.grb_model = grb_model
        self.idx = idx
        self.time_step = 0
        self.var = None
        self.constr = None

    def forward(self, prev_layer):
        self.time_step += 1


class InputLayer(Layer):
    def __init__(self, grb_model, inp_shape, lb=-GRB.INFINITY, ub=GRB.INFINITY, idx=0):
        super(InputLayer, self).__init__(grb_model, idx)
        self.inp_shape = inp_shape
        self.lb = lb
        self.ub = ub
        self.var = self.grb_model.addMVar(
        shape=inp_shape,
        vtype=GRB.CONTINUOUS,
        lb=lb,
        ub=ub,
        name=f"inp_var_time_{self.time_step}"
    )

    def forward(self, prev_layer=None):
        super(InputLayer, self).forward(prev_layer)


class Dense(Layer):
    def __init__(self, grb_model, kernel, bias, idx):
        super(Dense, self).__init__(grb_model, idx)
        self.kernel = kernel
        self.bias = bias
        self.units = bias.shape[0]

    def forward(self, prev_layer):
        super(Dense, self).forward(prev_layer)

        self.var = self.grb_model.addMVar(
            shape=self.units,
            vtype=GRB.CONTINUOUS,
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
            name=f"layer_{self.idx}_linear_var_time_{self.time_step}"
        )

        self.constr = self.grb_model.addConstr(
            self.var == prev_layer.var @ self.kernel + self.bias,
            name=f"layer_{self.idx}_linear_cnstr"
        )


class ReLU(Layer):
    def __init__(self, grb_model, idx):
        super(ReLU, self).__init__(grb_model, idx)
        self.units = None

    def forward(self, prev_layer):
        super(ReLU, self).forward(prev_layer)

        self.units = prev_layer.units
        self.var = self.grb_model.addMVar(
            shape=self.units,
            vtype=GRB.CONTINUOUS,
            lb=0,
            ub=GRB.INFINITY,
            name=f"layer_{self.idx}_relu_var_time_{self.time_step}"
        )

        for i in range(self.units):
            self.constr = self.grb_model.addConstr(
                self.var[i] == gp.max_(0, prev_layer.var[i]),
                name=f"layer_{self.idx}_relu_cnstr_{i}_time_{self.time_step}"
            )


class LeakyReLU(Layer):
    def __init__(self, grb_model, leakage, idx):
        super(LeakyReLU, self).__init__(grb_model, idx)
        self.leakage = leakage
        self.units = None

    def forward(self, prev_layer):
        super(LeakyReLU, self).forward(prev_layer)

        self.units = prev_layer.units
        self.var = self.grb_model.addMVar(
            shape=self.units,
            vtype=GRB.CONTINUOUS,
            lb=-GRB.INFINITY,
            ub=GRB.INFINITY,
            name=f"layer_{self.idx}_leaky_relu_var_time_{self.time_step}"
        )

        for i in range(self.units):
            self.constr = self.grb_model.addConstr(
                self.var[i] == gp.max_(self.leakage * prev_layer.var[i], prev_layer.var[i]),
                name=f"layer_{self.idx}_leaky_relu_cnstr_{i}_time_{self.time_step}"
            )
