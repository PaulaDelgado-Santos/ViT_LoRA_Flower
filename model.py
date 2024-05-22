from collections import OrderedDict

import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights
from transformers import ViTForImageClassification
from peft import get_peft_model, LoraConfig
from peft import get_peft_model_state_dict, set_peft_model_state_dict



def get_model():
    """Return a pretrained ViT with all layers frozen except output head."""

    # Instantiate a pre-trained ViT-B on ImageNet
    #model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
   

    # We're going to federated the finetuning of this model
    # using the Oxford Flowers-102 dataset. One easy way to achieve
    # this is by re-initializing the output block of the ViT so it
    # outputs 102 clases instead of the default 1k
    #in_features = model.heads[-1].in_features
    #model.heads[-1] = torch.nn.Linear(in_features, 102, bias=True)

    #peft_config = LoraConfig(
    #        r=8,
    #        lora_alpha=32,
    #        lora_dropout=0.01, #0.075
    #        target_modules=["out_proj"], #["query", "value"], #["out_proj"]
    #        task_type="FEATURE_EXTRACTOR",
    #    )

    ## Disable gradients for everything
    #model.requires_grad_(False)
    ## Now enable just for output head
    #model.heads.requires_grad_(True)

    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
    in_features = model.classifier.in_features
    model.classifier = torch.nn.Linear(in_features, 10)
    peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.01, #0.075
            target_modules=["query", "value"], #["out_proj"]
            task_type="FEATURE_EXTRACTOR",
        )

    return get_peft_model(model, peft_config)


def set_parameters(model, parameters):
    """Apply the parameters to the model.

    Recall this example only federates the head of the ViT so that's the only part of
    the model we need to load.
    """
    #finetune_layers = model.heads
    #state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    #params_dict = zip(finetune_layers.state_dict().keys(), parameters)
    #finetune_layers.load_state_dict(state_dict, strict=True)
    peft_state_dict_keys = get_peft_model_state_dict(model).keys()
    params_dict = zip(peft_state_dict_keys, parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    set_peft_model_state_dict(model, state_dict)


""" def train(net, trainloader, optimizer, epochs, device):
    ""Train the model on the training set.""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    avg_loss = 0
    # A very standard training loop for image classification
    for _ in range(epochs):
        for batch in trainloader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            avg_loss += loss.item() / labels.shape[0]
            loss.backward()
            optimizer.step()

    return avg_loss / len(trainloader) """

def train(net, trainloader, optimizer, epochs, device):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    avg_loss = 0
    # A very standard training loop for image classification
    for _ in range(epochs):
        for batch in trainloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs.logits, labels)
            avg_loss += loss.item() / labels.shape[0]
            loss.backward()
            optimizer.step()
    return avg_loss / len(trainloader)


""" def test(net, testloader, device: str):
    ""Validate the network on the entire test set.""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data["image"].to(device), data["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy
 """

def test(net, testloader, device: str):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data["img"].to(device), data["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs.logits, labels).item()
            _, predicted = torch.max(outputs.logits, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy
