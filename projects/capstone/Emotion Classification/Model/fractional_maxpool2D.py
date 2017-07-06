class FractionalMaxpool2D(Layer):
    def __init__(self, output_dim):
        super(FractionalMaxpool2D, self).__init__()
        self.output_dim = output_dim
    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        # This kind of layer doesn't have any variable
        pass
    def call(self, x):
        # Handle you algorithm here
        return ....
    def compute_output_shape(self, input_shape):
        # return the output shape
        return (input_shape[0], self.output_dim)
