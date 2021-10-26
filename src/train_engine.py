

def train_step(data_loader):
    for i, (images, labels) in enumerate(data_loader):
        print(images, labels)
        break