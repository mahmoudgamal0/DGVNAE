import torch

def save_model(epochs, model, optimizer, criterion, name, auc, ap):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    f = open('outputs/index.txt', 'a')
    f.writelines(f"Name: {name}, AP: {ap}, AUC: {auc}\n")
    
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, f'outputs/{name}.pth')