# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm.notebook import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(f"Using device: {device}")

# Define the custom model class
class ProteinPredictorWithESM(nn.Module):
    '''
    this class defines a custom ESM model with a single added layer encoding a per token attribution value
    '''
    def __init__(self, esm_model, esm_layer, hidden_dim=128):
        super(ProteinPredictorWithESM, self).__init__()
        self.esm_model = esm_model  # Pre-trained ESM-2 model
        self.esm_layer = esm_layer  # Layer from which to extract features
        self.fc1 = nn.Linear(self.esm_model.embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, batch_tokens, batch_lens):
        '''
        this function computes the updates for the model
        '''
        # Extract representations from ESM-2
        results = self.esm_model(batch_tokens, repr_layers=[self.esm_layer], return_contacts=False)
        token_representations = results["representations"][self.esm_layer]

        # Generate per-sequence representations via averaging (skip the first and last tokens)
        sequence_representations = []
        for i, tokens_len in enumerate(batch_lens):
            # Exclude special tokens (<cls> and <eos>)
            sequence_representation = token_representations[i, 1 : tokens_len - 1].mean(0)
            sequence_representations.append(sequence_representation)

        # Stack sequence representations into a tensor
        sequence_representations = torch.stack(sequence_representations).to(device)

        # Pass through fully connected layers with ReLU activation
        x = torch.relu(self.fc1(sequence_representations))
        x = self.fc2(x)  # Output layer (no activation, as we'll apply sigmoid later)
        return x

class ProteinDataset(Dataset):
    '''
    this class defines a custom dataset for a pytorch dataloader
    '''
    def __init__(self, data, batch_converter):
        """
        Initializes the dataset with protein data.

        Parameters:
        - data (list of tuples): Each tuple contains (protein_name, protein_sequence, label).
        - batch_converter (function): Function to convert batch data to tokens.
        """
        self.data = data  # List of (protein_name, protein_sequence, label) tuples
        self.batch_converter = batch_converter
        self.alphabet = batch_converter.alphabet  # Alphabet used for tokenization

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves the tokenized sequence and label for a given index.

        Returns:
        - batch_tokens (torch.LongTensor): Tokenized protein sequence.
        - batch_lens (torch.LongTensor): Length of the tokenized sequence.
        - label (torch.FloatTensor): Label (0 or 1).
        """
        protein_name, protein_sequence, label = self.data[idx]
        # Convert to batch format (name, sequence)
        batch_data = [(protein_name, protein_sequence)]
        # Convert to tokens using the batch converter
        _, _, batch_tokens = self.batch_converter(batch_data)
        # Compute length (number of tokens excluding padding)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)
        # Squeeze tensors to remove batch dimension
        batch_tokens = batch_tokens.squeeze(0).long()
        batch_lens = batch_lens.squeeze(0).long()
        return batch_tokens, batch_lens, torch.tensor([label], dtype=torch.float32)

def compute_attributions(batch_tokens, batch_lens, model, esm_model, esm_layer, device):
    """
    Compute attributions for each amino acid in the sequence.

    Parameters:
    - batch_tokens (torch.Tensor): Tokenized protein sequences [batch_size, seq_len]
    - batch_lens (torch.Tensor): Lengths of each protein sequence in the batch
    - model (torch.nn.Module): The trained model
    - esm_model (torch.nn.Module): The ESM model
    - esm_layer (int): The layer of the ESM model to use
    - device (torch.device): Device to run computations on

    Returns:
    - attributions (list of np.array): Attribution values for each amino acid
    """
    batch_tokens = batch_tokens.to(device)
    batch_lens = batch_lens.to(device)

    # Extract representations from ESM-2
    with torch.no_grad():
        results = esm_model(batch_tokens, repr_layers=[esm_layer], return_contacts=False)
        token_representations = results["representations"][esm_layer]

    # Generate per-sequence representations
    attributions = []
    for i, tokens_len in enumerate(batch_lens):
        tokens_len = tokens_len.item()
        # Exclude special tokens
        if tokens_len < 3:
            raise ValueError(f"Sequence length too short after excluding special tokens: {tokens_len}")
        # Exclude <cls> and <eos> tokens
        seq_repr = token_representations[i, 1:tokens_len - 1]  # [seq_len - 2, embed_dim]
        # Pass through model's layers
        x1 = torch.relu(model.fc1(seq_repr))  # [seq_len - 2, hidden_dim]
        x2 = model.fc2(x1)  # [seq_len - 2, 1]

        attributions.append(x2.squeeze(-1).detach().cpu().numpy())  # [seq_len - 2]
    return attributions

def makeDataLoader(df,batch_converter,colnames=['id','sequence','label'],shuffle=False):
    '''
    makes a data loader for a data frame with id, sequence, and (optionally) label for training or validation
    shuffle should be true for training and false for inference
    '''
    if(colnames[2] is not None and colnames[2] in df.columns):
        visualization_data = list(zip(df[colnames[0]], df[colnames[1]], df[colnames[2]]))
    else:
        df2=df.copy()
        df2['label']=np.nan
        visualization_data = list(zip(df2[colnames[0]], df2[colnames[1]], df2['label']))

    # Create dataset and data loader
    visualization_dataset = ProteinDataset(visualization_data, batch_converter)
    visualization_loader = DataLoader(
        visualization_dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False
    )
    return visualization_loader

def evaluateSeqs(dataloader,model,esm_model,esm_layer,device,maxlen=2000,verbose=True,usesigmoid=True):
    '''
    uses a fine-tuned esm2 model to predict a set of sequences from a data loader
    also return the attribution scores
    usesigmoid transforms the logits into a probability for binary predictions
    '''
    attr_outputs = []
    predictions = []
    ctr=0

    for batch_tokens, batch_lens, labels in dataloader:
        if(verbose):
            print('getting attributions for',ctr,'length',batch_lens)
        if(batch_lens[0]>maxlen):
            print('sequence too long, skipping')
            attr_outputs.append(np.array([]))
            predictions.append(np.nan)
            ctr+=1
            continue
        batch_tokens = batch_tokens.to(device)
        batch_lens = batch_lens.to(device)

        # Compute attributions
        attributions = compute_attributions(batch_tokens, batch_lens, model, esm_model, esm_layer, device)
        attr_outputs.append(attributions[0].flatten())

        # Get model predictions
        with torch.no_grad():
            outputs = model(batch_tokens, batch_lens)
            if(usesigmoid):
                outputs = torch.sigmoid(outputs)
            else:
                outputs=outputs.view(-1)
        predictions.append(outputs.cpu().numpy()[0].flatten()[0])
        ctr+=1
    return attr_outputs,predictions

def loadESMModel(nlayers=12,device='cpu'):
    '''
    load the esm model from the torch hub
    returns teh model, the representation layer, the batch converter, and the alphabet
    '''
    modeldict={12:"esm2_t12_35M_UR50D",30:"esm2_t30_150M_UR50D",
               33:"esm2_t33_650M_UR50D",36:"esm2_t36_3B_UR50D"}
    if(nlayers not in modeldict):
        print('unrecognized model size, aborting')
        return None
    print("Loading ESM-2 model...")
    esm_model, alphabet = torch.hub.load("facebookresearch/esm:main", modeldict[nlayers])
    esmlayer=nlayers
    # Freeze the ESM-2 model parameters to prevent training
    for param in esm_model.parameters():
        param.requires_grad = False

    # Move the model to the device (CPU or GPU)
    esm_model = esm_model.to(device)

    # Get the batch converter from the alphabet (list of amino acids)
    batch_converter = alphabet.get_batch_converter()

    print("ESM-2 model loaded.")
    return esm_model,esmlayer,batch_converter,alphabet

def runTraining(model,optimizer,criterion,train_loader,val_loader,num_epochs=15,regression=False):
    '''
    runs the training loop
    inputs are the model (ProteinPredictorWithESM), pytorch optimizer, pytorch criterion function, 
    pytorch training dataloader, pytorch validation dataloader, number of training epochs
    regression trains to predict a floating point value rather than a category
    accuracy for regression data is average of 1-|prediction-label|
    the output is lists of training loss, training accuracy, validation loss, and validation accuracy
    '''
    losses=[]
    accuracies=[]
    val_losses=[]
    val_accuracies=[]

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_loss = 0.0  # Initialize total loss for the epoch
        total_correct = 0.0

        for batch_tokens, batch_lens, labels in tqdm(train_loader):
            # Move data to device
            batch_tokens = batch_tokens.to(device)
            batch_lens = batch_lens.to(device)
            labels = labels.to(device)

            # Forward pass
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(batch_tokens, batch_lens)  # Get model outputs

            # Calculate loss and accuracy and backpropagate
            loss = criterion(outputs.view(-1), labels.view(-1))  # Compute loss
            if(not regression):
                prob = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities
                predlabels = (np.array(prob.cpu().detach().numpy()).flatten()>=0.5).astype(int)
                trainlabels = np.array(labels.cpu().detach().numpy()).flatten()
                amountcorrect=(predlabels==trainlabels).sum()
            else:
                prob=outputs.view(-1)
                predlabels = (np.array(prob.cpu().detach().numpy()).flatten()).astype(float)
                trainlabels = np.array(labels.cpu().detach().numpy()).flatten()
                amountcorrect=(1.0-np.abs(predlabels-trainlabels)).sum()

            loss.backward()  # Backpropagate the gradients
            optimizer.step()  # Update the model parameters

            total_loss += loss.item()  # Accumulate loss
            total_correct += amountcorrect

        val_loss=0
        val_correct=0
        for batch_tokens,batch_lens,labels in val_loader:
            # Move data to device
            batch_tokens = batch_tokens.to(device)
            batch_lens = batch_lens.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                outputs=model(batch_tokens,batch_lens)
            loss = criterion(outputs.view(-1), labels.view(-1))  # Compute loss
            if(not regression):
                prob = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities
                predlabels = (np.array(prob.cpu().detach().numpy()).flatten()>=0.5).astype(int)
                vallabels = np.array(labels.cpu().detach().numpy()).flatten()
                amountcorrect=(predlabels==vallabels).sum()
            else:
                prob=outputs.view(-1)
                predlabels = (np.array(prob.cpu().detach().numpy()).flatten()).astype(float)
                vallabels = np.array(labels.cpu().detach().numpy()).flatten()
                amountcorrect=(1.0-np.abs(predlabels-vallabels)).sum()
            val_loss+=loss.item()
            val_correct+=amountcorrect

        avg_loss = total_loss / len(train_loader)  # Calculate average loss for the epoch
        accuracy=total_correct / len(train_loader)
        val_avg_loss=val_loss/len(val_loader)
        val_acc=val_correct/len(val_loader)
        losses.append(avg_loss)
        accuracies.append(accuracy)
        val_losses.append(val_avg_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.3f}, Val_Loss: {val_avg_loss:.4f}, Val_Accuracy: {val_acc:.3f}")
    return losses,accuracies,val_losses,val_accuracies

def getESM2Representations(esm_model,batch_converter,seqs,rep_layer=12):
    '''
    gets the ESM2 representation matrices for the list of sequences
    inputs are the model, the batch converter, the position of the representation layer, and the list of sequences
    output is a list of representation layer matrices (
    '''
    # Extract per-residue representations
    # set the model to eval mode
    _=esm_model.eval()
    reprs=[]
    for i in tqdm(range(len(seqs))):
        data=[('tlabel',seqs[i])]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        with torch.no_grad():
            result = esm_model(batch_tokens.to(device), repr_layers=[rep_layer], return_contacts=False)
        reprs.append(result['representations'][rep_layer].cpu().numpy())
    return reprs
