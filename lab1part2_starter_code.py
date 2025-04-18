import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class StraightThroughEstimator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bits):
        abs_input = input.abs() #we need the abs_input in order to avoid mishandling of negative numbers
        sign = input.sign() #get the sign of the input and reserve one bit for this
        max_quantization_value = 2 ** (bits - 1) - 1 #e.g 3 bits --> 2^2 - 1 = 3 = 11(base 2) with one bit reserved for the sign bit
        max_input_value = abs_input.max() #we need the absolute value so that we can know the largest number to scale by such that we get this fake quantization effect
        quant_scale = max_input_value / max_quantization_value #the essential idea here is that you want the largest float value in the tensor to be scaled to the largest 8 bit value (or 6 bit)
        quantized_input_values = torch.round(abs_input/quant_scale).clamp(0, max_quantization_value)
        output = sign * quantized_input_values*quant_scale #dequantize it so the backprop can feel the quantization effects 
 
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # In the backward pass the gradients are returned directly without modification.
        # This is the key step of STE
        return grad_output, None

# To apply the STE
def apply_ste(x, bits):
    return StraightThroughEstimator.apply(x, bits)

# When you want to quantize the weights, call the apply_ste function
# You need to use this function within the forward pass of your model in a custom Conv2d class.

class QuantizedConv2d8(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True): 
        super(QuantizedConv2d8, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias )
    def forward(self, x):
          
  
        # x = self.conv(x)
        # x = apply_ste(x)
        # return x
        quantized_weight = apply_ste(self.conv.weight, bits=8)  
        return F.conv2d(x, quantized_weight, self.conv.bias, self.conv.stride, self.conv.padding)
    
class QuantizedConv2d6(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True): 
        super(QuantizedConv2d6, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias )
    def forward(self, x):
          
  
        # x = self.conv(x)
        # x = apply_ste(x)
        # return x
        quantized_weight = apply_ste(self.conv.weight, bits=6)  
        return F.conv2d(x, quantized_weight, self.conv.bias, self.conv.stride, self.conv.padding)

# To apply quantization and STE, you will need to define a new Conv2d class that will be used to replace the default Conv2d class in the ResNet model.
# You should follow a similar approach as Lab 1 Part 1.
Conv2dClass = QuantizedConv2d8 # this is for the 8 bit training 


# The code here is more complicated that it needs to be as it can be used to define multiple ResNet models with different configurations.
# You can ignore the specifics of the model code (contents of BasicBlock and ResNet classes), you only need to replace the Conv2dClass variable.
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2dClass(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = Conv2dClass(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Conv2dClass(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = Conv2dClass(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        
        self.linear = nn.Linear(64*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = F.avg_pool2d(out, out.size(2))  # Global Average Pooling. Unique to ResNet8.
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# ResNet8 variant
def ResNet8():
    return ResNet(BasicBlock, [1, 1, 1])

net = ResNet8().to(device)

# Hyperparameters
num_epochs = 50
batch_size = 128 # you can lower this to 64 or 32 to help speed up training.
learning_rate = 0.01

# Data transformations
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

# Initialize model, loss function and optimizer
model = ResNet8().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Training function
def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: [{epoch}][{batch_idx}/{len(trainloader)}] '
                  f'Loss: {train_loss/(batch_idx+1):.3f} '
                  f'Train Acc: {100.*correct/total:.3f}%')
# Testing function
def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    # Print summary on a new line after progress bar
    print(f'Epoch {epoch+1}: Test Acc: {100.*correct/total:.3f}%')
    return correct/total

'''number 1'''

Conv2dClass = nn.Conv2d
model = ResNet8().to(device)
model.load_state_dict(torch.load("lab1part2.pth", map_location=device))

#Run inference 
fp_accuracy = test(epoch=0)
print(f"Inference accuracy of pretrained FP network: {fp_accuracy*100:.2f}%")


'''number 2'''
Conv2dClass = QuantizedConv2d8
# Train the model
best_acc = 0
for epoch in range(num_epochs):
    train(epoch)
    acc = test(epoch)
    scheduler.step()
    
    # Save model if better than previous best
    if acc > best_acc:
        print(f'Saving model, acc: {acc:.3f} > best_acc: {best_acc:.3f}')
        best_acc = acc
        torch.save(model.state_dict(), 'lab1part2_qat.pth')

print(f'Best test accuracy: {best_acc*100:.2f}%')
print('Training completed! Model saved as lab1part2_qat.pth')
if(Conv2dClass == QuantizedConv2d8):
    eight_bit_accuracy = best_acc*100
elif(Conv2dClass == QuantizedConv2d6):
    six_bit_accuracy = best_acc*100
else: 
    print("conv2dclass = default model")


'''number 3'''
      
# Initialize model for 6 bits, loss function and optimizer
Conv2dClass = QuantizedConv2d6
model_6bit = ResNet8().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_6bit.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
best_acc = 0
for epoch in range(num_epochs):
    train(epoch)
    acc = test(epoch)
    scheduler.step()
    
    # Save model if better than previous best
    if acc > best_acc:
        print(f'Saving model, acc: {acc:.3f} > best_acc: {best_acc:.3f}')
        best_acc = acc
        torch.save(model.state_dict(), 'lab1part2_qat.pth')

print(f'Best test accuracy: {best_acc*100:.2f}%')
print('Training completed! Model saved as lab1part2_qat.pth')
if(Conv2dClass == QuantizedConv2d8):
    eight_bit_accuracy = best_acc*100
elif(Conv2dClass == QuantizedConv2d6):
    six_bit_accuracy = best_acc*100
else: 
    print("conv2dclass = default model")

# For grading, your QAT inference training script should include these *exact* two lines of code at the end:
print(f"Inference accuracy of pretrained FP network: {fp_accuracy*100:.2f}%")
print(f"8-bit accuracy:{eight_bit_accuracy}")
print(f"6-bit accuracy:{six_bit_accuracy}")