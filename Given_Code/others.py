import os
import torch
import csv

def read_csv_folder(folder_path):
    data = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            matrix = torch.tensor([[float(entry) for entry in row] for row in csv.reader(open(file_path))])
            data.append(matrix)
    return data

def calculate_overall_average(data):
    if len(data) == 0:
        return None
    overall_sum = sum(data)
    overall_average = overall_sum / len(data)
    return overall_average

def save_to_csv(overall_average, output_path):
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in overall_average.tolist():
            writer.writerow(row)


def train(model, loss_fn, device, data_loader, optimizer):
    """Performs an epoch of model training.

    Parameters:
    model (nn.Module): Model to be trained.
    loss_fn (nn.Module): Loss function for training.
    device (torch.Device): Device used for training.
    data_loader (torch.utils.data.DataLoader): Data loader containing all batches.
    optimizer (torch.optim.Optimizer): Optimizer used to update model.

    Returns:
    float: Total loss for epoch.
    """
    model.train()
    total_loss = 0

    for batch in data_loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        out= model(batch.x, batch.edge_index, batch.batch )  

        loss = loss_fn(out, batch.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss

def eval(model, device, loader):
    """Calculate accuracy for all examples in a DataLoader.

    Parameters:
    model (nn.Module): Model to be evaluated.
    device (torch.Device): Device used for evaluation.
    loader (torch.utils.data.DataLoader): DataLoader containing examples to test.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            pred = model(batch.x, batch.edge_index, batch.batch )  
            pred_class = torch.argmax(pred, dim=1)

            correct += (pred_class == batch.y).sum().item()
            total += batch.num_graphs

    accuracy = correct / total
    return accuracy


def train_plus(model, loss_fn, device, data_loader, optimizer):
    """Performs an epoch of model training.

    Parameters:
    model (nn.Module): Model to be trained.
    loss_fn (nn.Module): Loss function for training.
    device (torch.Device): Device used for training.
    data_loader (torch.utils.data.DataLoader): Data loader containing all batches.
    optimizer (torch.optim.Optimizer): Optimizer used to update model.

    Returns:
    float: Total loss for epoch.
    """
    model.train()
    total_loss = 0

    for batch in data_loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        out, _ = model(batch.x, batch.edge_index, batch.batch )  

        loss = loss_fn(out, batch.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss

def eval_plus(model, device, loader):
    """Calculate accuracy for all examples in a DataLoader.

    Parameters:
    model (nn.Module): Model to be evaluated.
    device (torch.Device): Device used for evaluation.
    loader (torch.utils.data.DataLoader): DataLoader containing examples to test.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            pred, _ = model(batch.x, batch.edge_index, batch.batch )  
            pred_class = torch.argmax(pred, dim=1)

            correct += (pred_class == batch.y).sum().item()
            total += batch.num_graphs

    accuracy = correct / total
    return accuracy



def clear_folder(directory):
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return
    
    files = os.listdir(directory)
    if not files:
        print(f"No files found in '{directory}'.")
    else:
        print(f"Deleting files in '{directory}':")
        for file_name in files:
            file_path = os.path.join(directory, file_name)
            os.remove(file_path)
            print(f"Deleted: {file_path}")


