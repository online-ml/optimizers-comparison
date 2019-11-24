import functools
from creme import base
from creme import compose
from creme import linear_model
from creme import metrics
from creme import optim
from creme import preprocessing
from creme import stream
from keras import backend as K
from keras import layers
from keras import models
from keras import optimizers
from sklearn import datasets
import torch


class PyTorchNet(torch.nn.Module):

    def __init__(self, n_features):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1)
        torch.nn.init.constant_(self.linear.weight, 0)
        torch.nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        return self.linear(x)


class PyTorchModel:

    def __init__(self, network, loss_fn, optimizer):
        self.network = network
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def fit_one(self, x, y):
        x = torch.FloatTensor(list(x.values()))
        y = torch.FloatTensor([y])

        y_pred = self.network(x)
        loss = self.loss_fn(y_pred, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return self


class PyTorchRegressor(PyTorchModel, base.Regressor):

    def predict_one(self, x):
        x = torch.FloatTensor(list(x.values()))
        return self.network(x).item()


class KerasModel:

    def __init__(self, model):
        self.model = model

    def fit_one(self, x, y):
        x = [[list(x.values())]]
        y = [[y]]
        self.model.train_on_batch(x, y)
        return self


class KerasRegressor(KerasModel, base.Regressor):

    def predict_one(self, x):
        x = [[list(x.values())]]
        return self.model.predict_on_batch(x)[0][0]


KERAS_EPS = K.epsilon()
LR = .01

OPTIMIZERS = {
    'SGD': (
        optim.SGD(lr=LR),
        functools.partial(torch.optim.SGD, lr=LR),
        optimizers.SGD(lr=LR)
    ),
    'Adam': (
        optim.Adam(lr=LR, beta_1=.9, beta_2=.999, eps=KERAS_EPS),
        functools.partial(torch.optim.Adam, lr=LR, betas=(.9, .999), eps=KERAS_EPS),
        optimizers.Adam(lr=LR, beta_1=.9, beta_2=.999)
    ),
    'AdaDelta': (
        optim.AdaDelta(rho=.95, eps=KERAS_EPS),
        functools.partial(torch.optim.Adadelta, rho=.95, eps=KERAS_EPS),
        optimizers.Adadelta(rho=.95)
    ),
    'AdaGrad': (
        optim.AdaGrad(lr=LR, eps=KERAS_EPS),
        functools.partial(torch.optim.Adagrad, lr=LR),
        optimizers.Adagrad(lr=LR)
    ),
    'Momentum': (
        optim.Momentum(lr=LR, rho=.1),
        functools.partial(torch.optim.SGD, lr=LR, momentum=.1),
        optimizers.SGD(lr=LR, momentum=.1)
    )
}


def add_intercept(x):
    return {**x, 'intercept': 1.}


for name, (creme_optim, torch_optim, keras_optim) in OPTIMIZERS.items():

    X_y = stream.iter_sklearn_dataset(
        dataset=datasets.load_boston(),
        shuffle=True,
        random_state=42
    )
    n_features = 13

    creme_lin_reg = (
        compose.FuncTransformer(add_intercept) |
        linear_model.LinearRegression(
            optimizer=creme_optim,
            l2=0,
            intercept_lr=0
        )
    )

    torch_model = PyTorchNet(n_features=n_features)
    torch_lin_reg = PyTorchRegressor(
        network=torch_model,
        loss_fn=torch.nn.MSELoss(),
        optimizer=torch_optim(torch_model.parameters())
    )

    inputs = layers.Input(shape=(n_features,))
    predictions = layers.Dense(1, kernel_initializer='zeros', bias_initializer='zeros')(inputs)
    keras_model = models.Model(inputs=inputs, outputs=predictions)
    keras_model.compile(optimizer=keras_optim, loss='mean_squared_error')
    keras_lin_reg = KerasRegressor(keras_model)

    creme_metric = metrics.MAE()
    torch_metric = metrics.MAE()
    keras_metric = metrics.MAE()

    scaler = preprocessing.StandardScaler()

    for x, y in X_y:

        x = scaler.fit_one(x).transform_one(x)

        creme_metric.update(y, creme_lin_reg.predict_one(x))
        creme_lin_reg.fit_one(x, y)

        torch_metric.update(y, torch_lin_reg.predict_one(x))
        torch_lin_reg.fit_one(x, y)

        keras_metric.update(y, keras_lin_reg.predict_one(x))
        keras_lin_reg.fit_one(x, y)

    print(name, creme_metric.get(), torch_metric.get(), keras_metric.get())
