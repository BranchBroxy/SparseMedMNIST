import torch
import torch.nn as nn
from get_data import add_noise_to_mnist_dataset

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
            model_runner.run()
            acc_list.append(model_runner.training_acc[-1][-1])

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
    plt.savefig("Compare.png")
    plt.close()




    print(acc_list_per_noise_level)

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
    from models import LinearNN, DenseModel, CNNModel, SparseModel

    dense_model = DenseModel(in_features=28 * 28, hidden_features=128, out_features=10, bias=True)
    cnn_model = CNNModel(in_features=28 * 28, hidden_features=128, out_features=10, bias=True)
    sparse_model = SparseModel(in_features=28 * 28, hidden_features=128, out_features=8, bias=True)
    from get_data import load_medmnist
    train_loader, test_loader = load_medmnist(batch_size=10, data_flag="tissuemnist")
    # compare_models_robustness(train_loader, test_loader, dense_model, sparse_model, cnn_model)
    from handle_model import handle_model
    sparse_model_runner = handle_model(sparse_model, train_loader, test_loader)
    sparse_model_runner.run()
    sparse_model_runner.plot_training_acc(save_string="sparse_training_acc_plot_.png")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
