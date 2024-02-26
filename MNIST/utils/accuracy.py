import torch

def calc_test_accy(model, test_loader, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    model.eval()   # Set model into evaluation mode
    correct = 0
    with torch.no_grad():
        for (data, target) in test_loader:
            data, target = data.to(device), target.to(
                device)  # Move data to GPU
            output = model(data)   # Calculate Output
            pred = output.max(1, keepdim=True)[1]  # Calculate Predictions
            correct += pred.eq(target.view_as(pred)).sum().item()

        return (100.*correct/len(test_loader.dataset))

def calc_train_accy(model, dataloader, num_batches, batch_size, device=torch.device("cuda:0" if torch.cuda.is_available()
                      else "cpu")):
    model.eval()
    correct = 0
    with torch.no_grad():
        data_iterator = iter(dataloader)
        for i in range(num_batches):  # iterate for the specified number of batches
            try:
                data, target = next(data_iterator)
            except StopIteration:
                data_iterator = iter(dataloader)
                data, target = next(data_iterator)

            data, target = data.to(device), target.to(
                device)  # Move data to GPU
            output = model(data)   # Calculate Output
            pred = output.max(1, keepdim=True)[1]  # Calculate Predictions
            correct += pred.eq(target.view_as(pred)).sum().item()

    return (100.*correct/(num_batches * batch_size))

def categorical_cross_entropy(y_true, y_pred):

    return torch.nn.NLLLoss()(torch.log(y_pred), y_true)

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round((y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc