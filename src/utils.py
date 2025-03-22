def save_model(model, filepath):
    import torch
    torch.save(model.state_dict(), filepath)

def load_model(model, filepath):
    import torch
    model.load_state_dict(torch.load(filepath))
    model.eval()

def preprocess_data(data):
    # 假設data是一個numpy陣列，這裡進行標準化處理
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    return (data - mean) / std

def create_directory(directory):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)