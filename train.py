from dataset import Ai4healthDataset
import torch
from model import network
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn   
import torch.nn.functional as F
from tqdm import tqdm
import param
from eval import eval

torch.backends.cudnn.benchmark = True  # Provides a speedup

args = param.parse_arguments()
torch.manual_seed(args.seed)

train_set = Ai4healthDataset(mode='train')
test_set = Ai4healthDataset(mode='test')
val_set = Ai4healthDataset(mode='val')

trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
valloader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

net = network.Ai4healthcareNet(args.backbone, args.fc_output_dim).to(args.device).train()
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(net.parameters(), lr=args.lr)

for epoch in tqdm(range(args.epochs_num)):
    avg_loss = 0
    for i, (data, labels, _) in enumerate(tqdm(trainloader)):
        data = data.to(args.device)
        labels = labels.to(args.device)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, labels)
        loss.backward() 
        optimizer.step()
        
        # loss
        avg_loss += loss / args.batch_size
    
    # evaluate on val_set
    recall_1, recall_5, recall_10, recall_20 = eval(args, trainloader, valloader, net)
    
    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_loss))
        
torch.save(net.state_dict(), 'Net_gem.pth')