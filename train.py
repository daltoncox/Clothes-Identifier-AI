import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import NeuralNetwork as NN

def best_available_device():
    device = (
        "cuda" if torch.cuda.is_available() # NVIDIA Cuda
        else "mps" if torch.backends.mps.is_available() # Apple Silicon
        else "cpu"
    )
    return device
def train_loop(device, dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    model.train() # Best practice: sets model to training mode
    for batch, (X, y) in enumerate(dataloader):
        X=X.to(device)
        y=y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward() # Calculate gradient 
        optimizer.step() # Take a step towards lower loss using the gradient
        optimizer.zero_grad() # Reset the gradient because by default it is cumulative

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(device, dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    with torch.no_grad():
        for X, y in dataloader: # X is image, y is the correct class
            X=X.to(device)
            y=y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Initialise datasets and training params
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
learning_rate = 1e-4
batch_size = 64
epochs = 10

# Initialize device and model
device = best_available_device()
model = NN.NeuralNetwork().to(device)
try:
    model.load_state_dict(torch.load('model/clothes_identifier.pt', weights_only=True))
    print("Model loaded successfully!")
except:
    print("Failed to load exsisting model. Training from scratch.")

#Initialize data loaders
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Initialize the loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(device, train_dataloader, model, loss_fn, optimizer)
    test_loop(device, test_dataloader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), 'model/clothes_identifier.pt')