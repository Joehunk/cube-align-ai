import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from generate_cuboid import normalize_point_cloud, denormalize_point_cloud
import time

device = torch.device('mps')

class PointNet(nn.Module):
    """
    v1
    """
    def __init__(self):
        super(PointNet, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # Output 3 numerical values
        )

    def forward(self, x):
        x = x.transpose(2, 1)  # Assuming input shape is [batch_size, num_points, 3]
        x = self.mlp1(x)
        x = torch.max(x, 2)[0]
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

class PointsDataset(Dataset):
    def __init__(self, file_path, normalize_fn):
        """
        Custom dataset for loading and normalizing XYZ coordinates, separating the features from the label.

        Parameters:
        file_path (str): Path to the .npz file containing the XYZ coordinates.
        normalize_fn (callable): Function to normalize the data. It should accept a NumPy array and return a normalized NumPy array.
        """
        data = np.load(file_path, allow_pickle=True)
        self.coordinates = []
        self.labels = []

        # Load, normalize, and separate features and labels
        for idx in range(len(data.files)):
            points_with_label_at_end = data[f'arr_{idx}']
            points_with_label_at_end_norm = normalize_fn(points_with_label_at_end)
            self.coordinates.append(points_with_label_at_end_norm[:-1, :])
            
            # Assuming the label is the last entry in the last dimension
            labels = points_with_label_at_end_norm[-1, :]
            self.labels.append(labels)

    def __len__(self):
        return len(self.coordinates)

    def __getitem__(self, idx):
        # Convert features and labels to PyTorch tensors
        coords_tensor = torch.tensor(self.coordinates[idx], dtype=torch.float32)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)  # Adjust dtype if labels are not float
        return coords_tensor, label_tensor
    
import torch

def random_sample_pad_collate(batch):
    """
    Pad point clouds in a batch to equal size by randomly sampling the smaller clouds.
    """
    # Find the largest point cloud size in the batch
    max_size = max([pc.size(0) for pc, _ in batch])  # point cloud is [num_points, 3]

    batch_padded = []
    for pc, label in batch:
        num_points = pc.size(0)  # Adjusted for [num_points, 3]
        if num_points < max_size:
            # Calculate the number of points needed for padding
            num_padding = max_size - num_points
            
            # Randomly sample indices from the point cloud
            indices = torch.randint(low=0, high=num_points, size=(num_padding,))
            
            # Use the indices to select points for padding
            padding_points = pc[indices, :]
            
            # Concatenate the original points with the padding points
            padded_pc = torch.cat([pc, padding_points], dim=0)
        else:
            padded_pc = pc
        
        batch_padded.append((padded_pc, label))
    
    # Stack all the padded point clouds and labels
    pcs = torch.stack([x[0] for x in batch_padded])
    labels = torch.stack([x[1] for x in batch_padded])
    return pcs, labels

    
def get_dataloader(file_name, batch_size=64):
    data_set = PointsDataset(file_name, lambda pc: normalize_point_cloud(pc)[0])
    return DataLoader(data_set, batch_size=batch_size, shuffle=True, collate_fn=random_sample_pad_collate)

# Example usage
pointnet = PointNet().to(device)
optimizer = optim.Adam(pointnet.parameters(), lr=0.001)
criterion = nn.MSELoss()  # Or another appropriate loss function

def train(num_epochs, train_loader):
    # Training loop
    for epoch in range(num_epochs):
        pointnet.train()
        train_loss = 0
        start_time = time.time()
        for batch_idx, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(device)
            labels = labels.to(device)
            output = pointnet(data)
            loss = criterion(output, labels)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        end_time = time.time()
        print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.3E} ({end_time - start_time}s)')

def test_train_loader(train_loader):
    for batch_idx, (data, labels) in enumerate(train_loader):
        print(f"data shape {data.shape}")
        print(f"label shape {labels.shape}")
        break

def predict(file_name):
    pointnet.eval()
    test_data = np.load(file_name)['arr_483']
    
    test_data_norm, centroid, max_dist = normalize_point_cloud(test_data)
    coords_norm = test_data_norm[:-1, :]
    label = test_data[-1, :]

    data = torch.tensor(coords_norm, dtype=torch.float32).unsqueeze(0).to(device)
    output_norm = pointnet(data).detach().cpu().numpy()
    output_denorm = denormalize_point_cloud(output_norm, centroid, max_dist)

    print(f"Label: {label} output: {output_denorm}")

def do_train():
    data = get_dataloader("./train.npz")
    train(8, data)
    torch.save(pointnet.state_dict(), './weights1_v1.pth')

if __name__ == "__main__":
    pointnet.load_state_dict(torch.load('./weights1_v1.pth'))
    predict('./test.npz')