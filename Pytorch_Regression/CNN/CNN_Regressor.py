class CNN_Regressor(nn.Module):
  def __init__(self, batch_size, inputs, outputs, kernel_size):
    # initialization of the superclass
    super(CNN_Regressor, self).__init__()
    # store the parameters
    self.batch_size = batch_size
    self.inputs = inputs
    self.outputs = outputs

    self.input_layer = Conv1d(inputs, batch_size, kernel_size=kernel_size, stride = 1)
    self.max_pooling_layer = MaxPool1d(kernel_size=kernel_size)
    self.conv_layer1 = Conv1d(batch_size, 128, kernel_size=kernel_size, stride=3)
    self.conv_layer2 = Conv1d(128, 256, kernel_size=kernel_size, stride=3)
    self.conv_layer3 = Conv1d(256, 512, kernel_size=kernel_size, stride=3)
    self.flatten_layer = Flatten()
    self.linear_layer = Linear(512, 128)
    self.output_layer = Linear(128, outputs)

  def feed(self, input):
    input = input.reshape((self.batch_size, self.inputs, 1))
    output = leaky_relu(self.input_layer(input))
 
    output = self.max_pooling_layer(output)
    output = leaky_relu(self.conv_layer1(output))

    output = self.max_pooling_layer(output)
    output = leaky_relu(self.conv_layer2(output))

    output = self.max_pooling_layer(output)
    output = leaky_relu(self.conv_layer3(output))

    output = self.flatten_layer(output)
    output = leaky_relu(self.linear_layer(output))

    output = self.output_layer(output)
    return output
