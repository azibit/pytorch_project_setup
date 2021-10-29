### Import Libraries

from __future__ import print_function # To bring the print function from Python 3 into Python 2.6+.

import csv, os, time, pickle, torch, shutil
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torchvision import models
from sklearn.metrics import classification_report

def move_file(filename, destination_dir):

    """
    Move file to destination directory

    filename: The path of file to be moved
    destination_dir: Directory to move image into
    """

    os.rename(filename, destination_dir + "/" + filename.split("/")[-1])

def delete_dir_if_exists(directory):
    """
    Remove a directory if it exists

    dir - Directory to remove
    """

    if os.path.exists(directory):
        shutil.rmtree(directory)

def create_dir(directory):
    """
    Create directory. Deletes and recreate directory if already exists

    Parameter:
    string - directory - name of the directory to create if it does not already exist
    """

    delete_dir_if_exists(directory)
    os.makedirs(directory)

def copy_files(file_list, destination_dir):

    """
    Copy files to destination directory

    file_list: The list of files to be moved
    destination_dir: Directory to move image into
    """

    for file in file_list:
        shutil.copy(file, destination_dir)


def get_transformation(with_augment):
    """
    Return a Baseline transformation with RandomCrop and Horizontal Flipping or simply returns a Normalized Tensor Transformation

    with_augment: Need the transformation with Cropping and Flipping
    """

    if with_augment:
        transformation = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
        ])
    else:
        transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])

    return transformation

def get_image_folder_dataset(dataset_path, transformation_required):
    """
    Return an ImageFolder based on the dataset_path and applied the transformation_required to each of the images

    dataset_path: The path where the dataset is located
    transformation_required: The transformation to apply to each of the images in the dataset
    """

    return datasets.ImageFolder(dataset_path, transformation_required)

def get_dataloader(dataset, batch_size, shuffle):
    """
    Returns the dataloader loading the dataset in the given batch_size and shuffles or not based on shuffle

    dataset: Determines the dataset to be loaded
    batch_size: Chooses the batch size with which to load the dataset
    shuffle: Shuffle the dataset items or not
    """

    return torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = shuffle)

def replace_fc_layer_for_resnets(net, number_of_classes):
    """
    Replace the last fully connected layer with new layer

    net: The model to be updated
    number_of_classes: The number of classes the new classifier would now have
    """

    num_features = net.fc.in_features

    net.fc = nn.Linear(num_features, number_of_classes)

    return net

def perform_prediction(model, inputs, targets, criterion_to_use):

    """
    Make a prediction with the given input

    model: Model to use to make prediction
    inputs: Input to make predictions on
    targets: Ground truth to compare predictions against
    criterion_to_use: The loss function to use
    """

    if torch.cuda.is_available():
        inputs, targets = inputs.cuda(), targets.cuda() # Load data into GPU

    #1. Make the model prediction
    outputs = model(inputs)

    #2. Calculate the loss based on the criterion_to_use
    loss = criterion_to_use(outputs, targets)

    #3. Count predictions
    _, predicted = torch.max(outputs.data, 1)
    correct_predictions = predicted.eq(targets.data).cpu().sum().float()

    return loss, correct_predictions, loss.data.item(), predicted

def normal_training_function(model, inputs, targets, criterion_to_use, optimizer):
    """
    Trains the model based on the given input, target and criterion to use

    model: The model to train
    inputs: The inputs to be trained on
    targets: The target to make comparison with the prediction
    criterion_to_use: The loss function to use
    optimizer: The optimizer function to use
    """

    #1. Perform the prediction
    loss, correct_predictions, train_loss, predicted = perform_prediction(model, inputs, targets, criterion_to_use)

    #2. Perform gradient descent and back propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return train_loss, correct_predictions

def mixup_data(inputs, targets, alpha = 1.0, use_cuda = True):
    """
    Mixes inputs and targets and returns a lambda value

    inputs: The given input
    targets: Ground truth for given inputs
    alpha: Value to determine to choose the lambda value
    use_cuda: Choose to use cuda or not
    """

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    #1. Get the size of the input
    batch_size = inputs.size()[0]

    if use_cuda:
        randomize_index = torch.randperm(batch_size).cuda()
    else:
        randomize_index = torch.randperm(batch_size)

    #2. Mixup the inputs with random images from other indices
    mixed_inputs = lam * inputs + (1 - lam) * inputs[randomize_index, :]

    #3. Get the labels based on the mixup
    targets_a, targets_b = targets, targets[randomize_index]

    return mixed_inputs, targets_a, targets_b, lam

def mixup_criterion(criterion, prediction, ground_truth_a, ground_truth_b, lam):
    return lam * criterion(prediction, ground_truth_a) + (1 - lam) * criterion(prediction, ground_truth_b)

def mixup_training_function(model, inputs, targets, criterion_to_use, optimizer):

    """
    Train the given input in the mixup fashion

    model: The model to train
    inputs: The inputs to be trained on
    targets: The target to make comparison with the prediction
    criterion_to_use: The loss function to use
    optimizer: The optimizer function to use
    """

    if torch.cuda.is_available():
        inputs, targets = inputs.cuda(), targets.cuda() # Load data into GPU

    #1. Mixup input
    mixed_inputs, targets_a, targets_b, lam = mixup_data(inputs, targets)

    #2. Make the model prediction
    outputs = model(inputs)

    #3. Compute the loss
    loss = mixup_criterion(criterion_to_use, outputs, targets_a, targets_b, lam)

    train_loss = loss.data.item()
    _, predicted = torch.max(outputs.data, 1)
    correct_predictions = (lam * predicted.eq(targets_a.data).cpu().sum().float() + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())

    #4. Perform gradient descent and back propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return train_loss, correct_predictions

def train_model(model, dataloader, epoch, training_function, criterion_to_use, optimizer_to_use):

    """
    Train a model with a given dataloader at a given epoch

    model: The model to be trained
    dataloader: The data to be loader
    epoch: Current epoch of training
    training_function: The training function to train on
    criterion_to_use: Criterion to use for training
    optimizer_to_use: Optimizer to use for training
    """

    model.train() # Set model to training mode

    train_loss = 0
    reg_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):

        # Train the current data batch
        loss, correct_predictions = training_function(model, inputs, targets, criterion_to_use, optimizer_to_use)

        # Update correct predictions
        correct += correct_predictions

        # Update total number of training examples
        total += targets.size(0)

        # Update the total loss
        train_loss += loss

    return correct, total, train_loss

def get_pred_as_list(test_preds):
    """
    Return a list of tensors as a list of numbers
    """

    result = []
    for i in range(len(test_preds)):
        result.append(int(test_preds[i].item()))

    return result

def test_model(model, dataloader, criterion_to_use, print_classification_report = False):
    """
    Test a given model

    model: The model to be tested
    dataloader: The dataloader to provide the data
    criterion_to_use: The criterion function to be used
    print_classification_report: Whether to print a classification report or not
    """

    model.eval()
    total = 0
    correct = 0

    training_loss = 0

    all_preds = torch.tensor([]).cuda()
    ground_truths = torch.tensor([]).cuda()

    for batch_idx, (inputs, targets) in enumerate(dataloader):

        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda() # Load data into GPU

        #1. Perform the prediction
        loss, correct_predictions, train_loss, predicted = perform_prediction(model, inputs, targets, criterion_to_use)

        #2. Update all model predictions and ground-truth values
        all_preds = torch.cat(
              (all_preds, predicted)
              ,dim=0
        )

        ground_truths = torch.cat(
              (ground_truths, targets)
              ,dim=0
        )

        #3. Update total number of training examples
        total += targets.size(0)

        #4. Update correct predictions
        correct += correct_predictions

        #5. Update Training loss
        training_loss += train_loss

    targets = get_pred_as_list(ground_truths)
    preds = get_pred_as_list(all_preds)

    if print_classification_report:
        return correct, total, training_loss, targets, preds

    return correct, total, training_loss

def adjust_learning_rate(optimizer, epoch, original_lr):
    """
    Decrease learning rate at 100 and 150 epochs

    optimizer: The optimizer with which to update learning rate
    epoch: Current training epoch
    original_lr: The original lr
    """

    lr = original_lr
    if epoch >= 100:
        lr /= 10

    if epoch >= 150:
        lr /= 10

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_model(model, path):
    """
    Source - https://pytorch.org/tutorials/beginner/saving_loading_models.html
    Save the model in the given path

    model: The Model to be saved
    path: Path to save the model
    """
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """
    Source - https://pytorch.org/tutorials/beginner/saving_loading_models.html
    Load the model from the given path

    model: The Model to be saved
    path: Path to save the model
    """

    model.load_state_dict(torch.load(path))

def plot_graph(train, val, epochs, label):
    """
    Source: # https://www.pluralsight.com/guides/data-visualization-deep-learning-model-using-matplotlib
    Plot a graph showing training and validation results

    train: The train list values
    val: The validation list values
    epochs: The total number of epochs for which results were collected
    label: The label of the graph
    """

    epochs = range(0, epochs)
    plt.plot(epochs, train, 'g', label='Training ' + label)
    plt.plot(epochs, val, 'b', label='Validation ' + label)
    plt.title('Training and Test ' + label)
    plt.xlabel('Epochs')
    plt.ylabel(label)
    plt.legend()
    plt.show()

def copy_to_other_dir(from_dir, to_dir):
    """
    Copy the content of the from directory into the to directory

    from_dir: Directory we are copying its content
    to_dir: Directory we are copying into
    """

    shutil.copytree(from_dir, to_dir)

def train(number_of_epochs, net, trainloader, training_function, criterion, optimizer, testloader, path_to_save, target_names):
    """
    Train a given model for the given number of epochs

    number_of_epochs: Number of epochs to run the model
    net: The model to be trained
    trainloader: Loader containing the training data
    training_function: The given training function to use for training
    criterion: Criterion for the loss
    optimizer: Optimizer
    testloader: Loader containing the test data
    path_to_save: Path to save model after training
    target_names: The list of the targets

    Return the training and test accuracy and loss list
    """

    train_loss_list = []
    train_acc_list = []

    test_loss_list = []
    test_acc_list = []

    best_acc = 0

    for epoch in range(0, number_of_epochs):

        print("... ", str(epoch), end = " ")

        #1. Train the model
        correct, total, train_loss = train_model(net, trainloader, epoch, training_function, criterion, optimizer)

        train_loss_list.append(train_loss)
        train_acc_list.append((correct / total).data.item() * 100)

        #2. Test the model
        if epoch == number_of_epochs - 1:
            correct, total, test_loss, targets, preds = test_model(net, trainloader, criterion, True)
            print("Train Result")
            print(classification_report(targets, preds, target_names=target_names))
            
            correct, total, test_loss, targets, preds = test_model(net, testloader, criterion, True)
            print("Test Result")
            print(classification_report(targets, preds, target_names=target_names))

        else:
            correct, total, test_loss = test_model(net, testloader, criterion)

        test_loss_list.append(test_loss)

        test_acc = (correct / total).data.item() * 100
        test_acc_list.append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc

            # Save the final model
            save_model(net, path_to_save)

    print("**********")
    return train_loss_list, test_loss_list, train_acc_list, test_acc_list
