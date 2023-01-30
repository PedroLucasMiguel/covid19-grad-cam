import hp
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import transforms
from torchvision import datasets
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from densenet201.densenet201 import get_model

def reset_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()

# Definindo as transformações que precisam ser feitas no conjunto de imagens
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Carregando o dataset a partir da pasta
dataset = datasets.ImageFolder("../dataset/", preprocess)

# Criando o K-Fold cross validation
kfold = KFold(n_splits=hp.K_FOLDS, shuffle=True)

# Melhor resultado da métrica F1 (utilizado no processo de checkpoint)
best_f1 = 0
best_f1_file_name = ""

# Dicionário que guarda todas as métricas para que possamos exportar em um json
metrics_json = {}

for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
    print(f"Doing for fold - {fold}")

    metrics_json[fold] = {}

    # Criando os "samplers" que vão aleatorizar a escolha das imagens
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

    # Criando os "loaders" para o nosso conjunto de treino e validação
    trainloader = torch.utils.data.DataLoader(
                      dataset, 
                      batch_size=hp.BATCH_SIZE, sampler=train_subsampler)
    testloader = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=hp.BATCH_SIZE, sampler=test_subsampler)

    # Utiliza GPU caso possível
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} for training")

    # Cria o modelo, reseta os pesos e define o dispositivo de execução
    model = get_model(True, False)
    model.apply(reset_weights)
    model.to(device)

    # Definindo nossa função para o calculo de loss e o otimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), momentum=hp.MOMENTUM, weight_decay=hp.WEIGHT_DECAY, lr=hp.LEARNING_RATE)

    # Iniciando o processo de treinamento
    for epoch in range(0, hp.EPOCHS):
        print(f"Doing for epoch - {epoch}")
        
        print("Training...")
        for i, data in enumerate(trainloader, 0):
            img, label = data
            img = img.to(device)
            label = label.to(device)
            model.train()
            pred = model(img)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        accuracy = []
        precision = []
        recall = []
        f1 = []
        model.eval()

        print("Validating...")
        # Iniciando o processo de validação
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                img, label = data
                img = img.to(device)
                label = label.to(device)
                outputs = model(img)
                _, predicted = torch.max(outputs.data, 1)
                # Convertendo os arrays das labels e das previsões para uso em CPU
                label_cpu = label.cpu()
                predicted_cpu = predicted.cpu()
                # Calculando as métricas
                precision.append(precision_score(label_cpu, predicted_cpu))
                recall.append(recall_score(label_cpu, predicted_cpu))
                f1.append(f1_score(label_cpu, predicted_cpu))
                accuracy.append(accuracy_score(label_cpu, predicted_cpu))

        # Apresentando as métricas
        accuracy = np.mean(accuracy)
        precision = np.mean(precision)
        recall = np.mean(recall)
        f1 = np.mean(f1)
        print("\nResults:")
        print(f"Accuracy for fold {fold}: {100.0 * accuracy} %%")
        print(f"Precision {(precision)}")
        print(f"Recall {recall}")
        print(f"F1 {f1}")
        
        # Se o resultado possuir a melhor medida F de todo o processo, salve o treinamento
        if f1 > best_f1:
            if best_f1_file_name != "":
                os.remove(f"./checkpoints/{best_f1_file_name}")

            print(f"\nSaving saving training for Fold={fold} and Epoch={epoch}")
            torch.save(model.state_dict(), f"./checkpoints/f_{fold}_e_{epoch}_savestate.pt")
            best_f1_file_name = f"f_{fold}_e_{epoch}_savestate.pt"

        metrics_json[fold][epoch] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        }
        print('--------------------------------')

# Exporta as métricas em um arquivo .json
with open("metrics.json", "w") as json_file:
    json.dump(metrics_json, json_file)
