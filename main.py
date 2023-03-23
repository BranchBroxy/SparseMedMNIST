import torch
import torch.nn as nn
from get_data import add_noise_to_mnist_dataset


def init_function():
    # Use a breakpoint in the code line below to debug your script.
    print(torch.cuda.current_device())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

def compare_models_robustness(train_dataloader, test_dataloader, *models: nn.Module, noise_levels=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]) -> None:
    from handle_model import handle_model
    acc_list_per_noise_level = []
    # noise_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for noise_level in noise_levels:
        print(noise_level)
        noisy_train_dataset = add_noise_to_mnist_dataset(train_dataloader.dataset, noise_level=noise_level)
        noisy_train_loader = torch.utils.data.DataLoader(dataset=noisy_train_dataset, batch_size=10, shuffle=False)

        noisy_test_dataset = add_noise_to_mnist_dataset(test_dataloader.dataset, noise_level=noise_level)
        noisy_test_loader = torch.utils.data.DataLoader(dataset=noisy_test_dataset, batch_size=10, shuffle=False)

        model_handlers = []
        model_names = ["Noise Level"]
        for model in models:
            model_handlers.append(handle_model(model, noisy_train_loader, noisy_test_loader))
            model_names.append(model.__class__.__name__)
            #models[0].__class__.__name__
        acc_list = []
        for model_runner in model_handlers:
            model_runner.run(epochs=2)
            acc_list.append(model_runner.training_acc_with_epoch[-1][-1])

        list_to_append = []
        list_to_append.append(noise_level)
        for acc in acc_list:
            list_to_append.append(acc)

        acc_list_per_noise_level.append(list_to_append)

    import pandas as pd
    df = pd.DataFrame(acc_list_per_noise_level)

    df.columns = model_names
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    figure, axes = plt.subplots()
    axes.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    plt.grid()
    axes.xaxis.set_major_formatter(ticker.ScalarFormatter())
    plt.ylabel("Accuracy in %")
    plt.xlabel("Noise Level")
    plt.title("Accuracy comparison of NN with MNIST")
        # df = pd.DataFrame(data=acc_list, columns=["Epoch", "Accuracy"])

    for index, row in df.iterrows():
        print(row)
    for i in range(df.shape[1]-1):
        sns.lineplot(data=df, x=df.iloc[:, 0], y=df.iloc[:, i+1], ax=axes, label=df.columns[i+1], marker="*",
                     markersize=8)

    plt.legend(loc='lower right')
        # axes.legend(labels=["Acc1", "Acc2"])
    plt.savefig("Compare_noise_mnist.png")
    plt.close()

    print(acc_list_per_noise_level)
def compare_models_acc_over_epoch(train_dataloader, test_dataloader, *models: nn.Module, path_save="Compare_acc.png") -> None:
    from handle_model import handle_model
    acc_list_per_noise_level = []
    model_handlers = []
    model_names = []
    for model in models:
        model_handlers.append(handle_model(model, train_dataloader, test_dataloader))
        model_names.append(model.__class__.__name__)
        #models[0].__class__.__name__
    acc_list = []
    for model_runner in model_handlers:
        model_runner.run(epochs=10)
        acc_list.append(model_runner.training_acc)



    import pandas as pd
    df = pd.DataFrame(acc_list).T

    df.columns = model_names
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
        # df = pd.DataFrame(data=acc_list, columns=["Epoch", "Accuracy"])

    #for index, row in df.iterrows():
       # print(row)
    for i in range(df.shape[1]):
        sns.lineplot(data=df, x=df.index+1, y=df.iloc[:, i], ax=axes, label=df.columns[i], marker="*",
                     markersize=8)

    plt.legend(loc='lower right')
        # axes.legend(labels=["Acc1", "Acc2"])
    plt.savefig(path_save)
    plt.close()

def med_data():
    import torch
    import torchvision
    from handle_model import handle_model
    from models import LinearNN, DenseModel, CNNModel, SparseModel
    from get_data import load_medmnist
    dense_model = DenseModel(in_features=28 * 28, hidden_features=128, out_features=10, bias=True)
    cnn_model = CNNModel(in_features=28 * 28, hidden_features=128, out_features=10, bias=True)
    sparse_model = SparseModel(in_features=28 * 28, hidden_features=128, out_features=8, bias=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = load_medmnist(batch_size=10, data_flag="tissuemnist")
    sparse_model_runner = handle_model(sparse_model, train_loader, test_loader)
    sparse_model_runner.run()
    #sparse_model_runner.plot_training_acc(save_string="sparse_training_acc_plot.png")




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # med_data()
    init_function()
    from models import LinearNN, DenseModel, CNNModel, SparseModel, SparseNNModel, SelfConnectedSparseModel, HNN, HNNV2, HNNV3

    # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

    from resnet_model import ResNet, ResidualBlock

    from get_data import load_mnist, load_medmnist
    train_loader, test_loader = load_medmnist(batch_size=10, data_flag="octmnist")
    dataset_shape = train_loader.dataset.imgs.shape[1:3]
    flatten_image_shape = dataset_shape[0] * dataset_shape[1]
    out_features = 4


    resnet = ResNet(ResidualBlock, [2, 2, 2, 2])
    dense_model = DenseModel(in_features=flatten_image_shape, hidden_features=512, out_features=out_features, bias=True)
    cnn_model = CNNModel(in_features=flatten_image_shape, hidden_features=512, out_features=out_features, bias=True)
    sparse_model = SparseModel(in_features=flatten_image_shape, hidden_features=512, out_features=out_features, bias=True)
    sparse_nn_model = SparseNNModel(in_features=flatten_image_shape, hidden_features=512, out_features=out_features, bias=True)
    self_connected_sparse_model = SelfConnectedSparseModel(in_features=flatten_image_shape, hidden_features=512, out_features=out_features, bias=True)
    hnn = HNN(in_features=flatten_image_shape, hidden_features=512, out_features=out_features, bias=True)
    hnnv2 = HNNV2(in_features=flatten_image_shape, hidden_features=512, out_features=out_features, bias=True)
    hnnv3 = HNNV3(in_features=flatten_image_shape, hidden_features=512, out_features=out_features, bias=True)

    # train_loader, test_loader = load_mnist(batch_size=10)


    compare_models_acc_over_epoch(train_loader, test_loader, resnet, hnnv3, hnnv2, dense_model, cnn_model)
    compare_models_robustness(train_loader, test_loader,hnnv3, hnnv2, hnn, dense_model, cnn_model, sparse_model, sparse_nn_model, self_connected_sparse_model)

    from handle_model import handle_model
    #sparse_model_runner = handle_model(sparse_nn_model, train_loader, test_loader)
    #sparse_model_runner.run()
    #sparse_model_runner.plot_training_acc(save_string="sparse_training_acc_plot_.png")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
