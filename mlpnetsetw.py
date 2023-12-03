import torch


def mlpnetsetw(custom_mlp, w):
    count = 0
    for layer in custom_mlp.layers:
        # Calculate the number of weights and biases for the current layer
        input_size, output_size = layer.w.shape[1], layer.w.shape[0]
        num_weights = input_size * output_size
        num_biases = output_size

        # Reshape and set the weights
        layer_weights = w[count:count +
                          num_weights].reshape(output_size, input_size)
        layer.w.data = torch.tensor(layer_weights, dtype=layer.w.data.dtype)
        count += num_weights

        # Set the biases
        layer_biases = w[count:count + num_biases].reshape(-1, 1)
        layer.b.data = torch.tensor(layer_biases, dtype=layer.b.data.dtype)
        count += num_biases

    return custom_mlp
