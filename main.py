# step 1 - setup and feature extraction
import torch
import torchvision.transforms as transforms, torchvision
from torchvision.models.resnet import resnet18
import torchvision
import torchvision.transforms as transforms
from simclr import SimCLR
from simclr.modules import NT_Xent



# augmentations will need to be done here
# random resized crops - The algorithm takes a random section of the image
#  and resizes it back to the standard 32x32 pixel size
# Random horizontal flips
# Color jittering
# Random grayscaling


class ContrastiveTransformations(object):

    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]

base_transform = transforms.Compose([
    
    transforms.RandomResizedCrop(size=32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.5, 0.5, 0.5, 0.2),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor()
])



print("Downloading CIFAR 10")
train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=ContrastiveTransformations(base_transform)
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=512,
    shuffle=True,
    drop_last=True
)


encoder = resnet18()
n_features = encoder.fc.in_features
model = SimCLR(encoder, projection_dim=128, n_features=n_features)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimiser = torch.optim.SGD(
    model.parameters(),
    lr=0.4,
    momentum=0.9,
    weight_decay=0.0001
)

criterion = NT_Xent(batch_size=512, temperature=0.5, world_size=1)
epochs = 1


for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch_idx, (images, _) in enumerate(train_loader):
        view1 = images[0].to(device)
        view2 = images[1].to(device)
        
        # 1. Wipe the old math memory
        optimiser.zero_grad()
        hi, hj, zi, zj = model(view1, view2)

        loss = criterion(zi, zj)

        loss.backward()

        optimiser.step()

        total_loss+= loss.item()

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")


# stage 2 - k means clustering alogirhtm 


