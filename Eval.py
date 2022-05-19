from MNIST import Net
import torch
from Networks import Generator
import numpy as np
from torchvision.transforms.functional import normalize

device = torch.device("cuda:0")

def generate_samples(generator, latent_size, num_image=26000, batch_size=100):
    data_sample = torch.empty((num_image, 3, 28, 28))

    for i in range(0, num_image, batch_size):
        start = i
        end = i + batch_size
        z = torch.randn(batch_size, latent_size, device=device)
        d = generator(z)
        data_sample[start:end] = d.cpu().data[:,:,2:30,2:30]
    
    return data_sample

def compute_scores(data, classifier):
    targets = np.zeros(1000, dtype=np.int32)
    for i in range(len(data)):
        y = np.zeros(3, dtype=np.int32)
        for j in range(3):#R, G, B
            x = data[i, j, :, :]
            x = torch.unsqueeze(x, dim=0)
            x = torch.unsqueeze(x, dim=0)
            x = x.to(device)
            
            x = (x + 1.) / 2.
            x = normalize(x.view(1, 28, 28), (0.1307,), (0.3081,)).view(1, 1, 28, 28)
            
            output = classifier(x)
            predict = output.cpu().detach().max(1)[1]
            y[j] = predict
        result = 100 * y[0] + 10 * y[1] + y[2]
        targets[result] += 1
    
    covered_targets = np.sum(targets != 0)
    Kl_score = 0
    for i in range(1000):
        if targets[i] != 0:
            q = targets[i] / len(data)
            Kl_score +=  q * np.log(q * 1000)
    return covered_targets, Kl_score

model = torch.load('epoch_100.pth', map_location='cpu')

nz = 512
G = Generator(NoiseDimension=nz, LatentMappingDepth=2, StageWidths=[1024, 512, 512], BlocksPerStage=[1, 1, 1]).to(device).eval()
G.load_state_dict(model['g_ema_state_dict'], strict=True)

classifier = Net().to(device).eval()
classifier.load_state_dict(torch.load('mnist_cnn.pt', map_location='cpu'))

data = generate_samples(G, nz)
covered_targets, Kl_score = compute_scores(data, classifier)
print('mode coverage: ' + str(covered_targets))
print('KLD: ' + str(Kl_score))










## validate dataset
## mode coverage: 1000
## KLD: 0.008448338779849796

# classifier = Net().to(device).eval()
# classifier.load_state_dict(torch.load('mnist_cnn.pt'))

# data = torch.load('./Data/data.pt')[:,:,2:30,2:30].to(device)
# covered_targets, Kl_score = compute_scores(data, classifier)
# print('mode coverage: ' + str(covered_targets))
# print('KLD: ' + str(Kl_score))