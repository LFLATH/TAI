def predict(network, input):
    output = input
    for layer in network:
        output = layer.forwardfeed(output)
    return output

def train(network, loss, loss_p, x_train, y_train, epochs = 1000, learning_pace = 0.01):
    for epoch in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
        
            output = predict(network, x)
            #print(output)
            error += loss(y, output)

            grad = loss_p(y, output)
            for layer in reversed(network):
                grad = layer.backwardprop(grad, learning_pace)

        error /= len(x_train)

        print(f"{epoch + 1}/{epochs}, error={error}")
    