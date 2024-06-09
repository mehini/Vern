from keras import layers, models, datasets
import gurobipy as gp
from gurobipy import GRB
from src import InputLayer, Dense, ReLU, VModel

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# Parameters
inp_shape = 28 * 28
num_classes = 10

# Preprocess the data
x_train = x_train.reshape((60000, inp_shape))
x_train = x_train.astype('float32') / 255

x_test = x_test.reshape((10000, inp_shape))
x_test = x_test.astype('float32') / 255

# Define the model
model = models.Sequential([
    layers.Input(shape=(28 * 28,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adamw',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
# model.fit(x_train, y_train, epochs=5, batch_size=500)

# Model directory
resources_dir = 'resources/models/'
model_dir = resources_dir + 'mnist_64_relu_10.weights.h5'

# Save the model
# model.save_weights(model_dir)

# Load the model
model.load_weights(model_dir)

# Extract the weights
kernels = []
biases = []
for layer in model.layers:
    kernels.append(layer.get_weights()[0])
    biases.append(layer.get_weights()[1])

# Test point
idx = 0
test_point = x_test[idx]
epsilon = 0.01
lb, ub = test_point - epsilon, test_point + epsilon

# Build a Gurobi model
grb_model = gp.Model("model")

# The input layer
input_layer = InputLayer(grb_model, inp_shape, lb, ub)

# Add the input constraint
grb_model.addConstr(input_layer.var >= lb, name="inp_lb")
grb_model.addConstr(input_layer.var <= ub, name="inp_ub")

# The rest of the layers
layer_1 = Dense(grb_model, kernels[0], biases[0], 1)
layer_2 = ReLU(grb_model, 2)
layer_3 = Dense(grb_model, kernels[1], biases[1], 3)

# The model
vmodel = VModel([input_layer, layer_1, layer_2, layer_3])
vmodel.forward()

# Compute the min and max for all logits
min_logits = []
max_logits = []

# Compute the output bounds
for i in range(num_classes):
    # Lower bound
    grb_model.setObjective(vmodel.layers[-1].var[i], GRB.MINIMIZE)
    grb_model.optimize()
    min_logits.append(grb_model.objVal)

    # Upper bound
    grb_model.setObjective(vmodel.layers[-1].var[i], GRB.MAXIMIZE)
    grb_model.optimize()
    max_logits.append(grb_model.objVal)

# Print the bounds
print("Min logits: ", min_logits)
print("Max logits: ", max_logits)
