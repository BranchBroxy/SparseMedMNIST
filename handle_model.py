import torch
import torch.nn as nn

class handle_model():
    """
    A class for handling PyTorch model training and testing on provided datasets.

    Args:
    model: A PyTorch model object to train and test.
    train_dataloader: A PyTorch DataLoader object containing training data.
    test_dataloader: A PyTorch DataLoader object containing testing data.

    Attributes:
    model: A PyTorch model object to train and test.
    train_dataloader: A PyTorch DataLoader object containing training data.
    test_dataloader: A PyTorch DataLoader object containing testing data.
    epochs: An integer representing the number of epochs to train the model.

    Methods:
    run(): Runs the model training and testing on provided datasets.
    train(dataloader, model, loss_fn, optimizer, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')): Trains the model on the provided training dataset.
    test(dataloader, model, loss_fn, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')): Tests the trained model on the provided testing dataset.
    """
    def __init__(self, model, train_dataloader, test_dataloader):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = model
        self.training_acc = []
        self.training_loss = []
        self.name_of_model = model.__class__.__name__

    def run(self):
        """
        Trains and tests the model on provided datasets for specified number of epochs.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model = LinearNN().to(device)
        self.model.to(device)
        self.loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.epochs = 5
        print(f"Running model")
        for self.epoch in range(self.epochs):
            print(f"Epoch {self.epoch + 1}\n-------------------------------")
            self.train(self.train_dataloader, self.model, self.loss_fn, optimizer)
            self.test(self.test_dataloader, self.model, self.loss_fn)

        print("Done!")
    def train(self, dataloader, model, loss_fn, optimizer, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """
        Trains the provided PyTorch model on the provided training dataset.

        Args:
        dataloader: A PyTorch DataLoader object containing training data.
        model: A PyTorch model object to train.
        loss_fn: A PyTorch loss function.
        optimizer: A PyTorch optimizer.
        device: A string representing the device on which the training should be performed.

        Returns:
        None
        """
        import torch.nn.functional as F
        size = len(dataloader.dataset)
        n_total_steps = len(dataloader)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            #max_indices = pred.argmax(dim=1)
            #max_tensor = max_indices

            y = y.squeeze(1)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch + 1) % 1000 == 0:
                print(f'Epoch [{self.epoch + 1}/{ self.epochs}], Step [{batch + 1}/{n_total_steps}], Loss: {loss.item():.4f}')
                #print(self.model.sl1.weight)
                #print(model.sl2.weight)
                #print(model.sl2.connections)


    def test(self, dataloader, model, loss_fn, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """
        Computes the accuracy and average loss of the model on a test dataset.
        Parameters:
        -----------
        dataloader : torch.utils.data.DataLoader
            A DataLoader for the test dataset.
        model : torch.nn.Module
            The trained model.
        loss_fn : torch.nn.Module
            The loss function for the model.
        device : str or torch.device, optional (default: 'cuda' if available, else 'cpu')
            The device on which to run the model.

        Returns:
        --------
        None
        """
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                y = y.squeeze(1)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        self.training_acc.append([self.epoch+1, 100*correct]),

    def plot_training_acc(self, save_string="training_acc_plot.png"):
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        figure, axes = plt.subplots()
        axes.xaxis.set_major_locator(ticker.MultipleLocator(1))
        plt.grid()
        axes.xaxis.set_major_formatter(ticker.ScalarFormatter())
        plt.ylabel("Accuracy in %")
        plt.xlabel("Epochs")
        plt.title("Accuracy comparison of NN with MNIST")
        df = pd.DataFrame(data=self.training_acc, columns=["Epoch", "Accuracy"])
        sns.lineplot(data=df, x=df.Epoch, y=df.Accuracy, ax=axes)
        plt.savefig(save_string)
        plt.close()
