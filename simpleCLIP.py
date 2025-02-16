import json
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import nltk
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Download NLTK tokenizer data if needed:
# nltk.download('punkt')

class COCODataset(Dataset):
    def __init__(self, images_dir, captions_file, transform=None, tokenizer=None, max_seq_length=20):
        """
        images_dir: directory where COCO train2017 images are stored.
        captions_file: path to captions_train2017.json.
        transform: image transformations.
        tokenizer: function to tokenize captions.
        max_seq_length: maximum length for tokenized captions.
        """
        self.images_dir = images_dir
        self.transform = transform if transform is not None else transforms.ToTensor()
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer if tokenizer is not None else nltk.word_tokenize

        # Load captions JSON
        with open(captions_file, 'r') as f:
            data = json.load(f)
        # Create a mapping from image_id to file_name
        self.image_id_to_file = {img['id']: img['file_name'] for img in data['images']}
        # Create a list of (image_id, caption) pairs; using only one caption per image (could randomize)
        self.samples = []
        for ann in data['annotations']:
            img_id = ann['image_id']
            caption = ann['caption']
            # Make sure the image file exists
            if img_id in self.image_id_to_file:
                self.samples.append((img_id, caption))
        print(f"Loaded {len(self.samples)} samples.")

        # Build a vocabulary (for simplicity, a very basic version)
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.build_vocab()

    def build_vocab(self):
        idx = len(self.vocab)
        for _, caption in self.samples:
            tokens = self.tokenizer(caption.lower())
            for token in tokens:
                if token not in self.vocab:
                    self.vocab[token] = idx
                    idx += 1

    def tokenize_caption(self, caption):
        tokens = self.tokenizer(caption.lower())
        # Convert tokens to indices
        indices = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
        # Pad or truncate to max_seq_length
        if len(indices) < self.max_seq_length:
            indices += [self.vocab["<PAD>"]] * (self.max_seq_length - len(indices))
        else:
            indices = indices[:self.max_seq_length]
        return torch.tensor(indices)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, caption = self.samples[idx]
        file_name = self.image_id_to_file[img_id]
        image_path = os.path.join(self.images_dir, file_name)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        caption_indices = self.tokenize_caption(caption)
        return image, caption_indices

# A simple image encoder: a small CNN that outputs a feature vector.
class SimpleImageEncoder(nn.Module):
    def __init__(self, output_dim=256):
        super(SimpleImageEncoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64, output_dim)
        
    def forward(self, x):
        x = self.conv(x)        # (batch, 64, 1, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# A simple text encoder: a GRU-based model that turns a sequence of tokens into a feature vector.
class SimpleTextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256, output_dim=256):
        super(SimpleTextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x: (batch, seq_length)
        x = self.embedding(x)          # (batch, seq_length, embed_dim)
        _, h = self.gru(x)             # h: (1, batch, hidden_dim)
        h = h.squeeze(0)               # (batch, hidden_dim)
        h = self.fc(h)                 # (batch, output_dim)
        return h

# Normalize vectors along the feature dimension.
def l2_normalize(x, eps=1e-8):
    return x / (x.norm(dim=1, keepdim=True) + eps)

# Contrastive loss similar to InfoNCE.
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, image_embeds, text_embeds):
        # Normalize embeddings
        image_embeds = l2_normalize(image_embeds)
        text_embeds = l2_normalize(text_embeds)
        # Compute similarity matrix (batch x batch)
        logits = torch.matmul(image_embeds, text_embeds.t()) / self.temperature
        batch_size = logits.shape[0]
        labels = torch.arange(batch_size).to(logits.device)
        # Compute symmetric cross-entropy loss
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.t(), labels)
        loss = (loss_i + loss_t) / 2
        return loss

# Training function for one epoch
def train_epoch(image_encoder, text_encoder, dataloader, optimizer, criterion, device):
    image_encoder.train()
    text_encoder.train()
    running_loss = 0.0
    for images, texts in dataloader:
        images = images.to(device)
        texts = texts.to(device)
        optimizer.zero_grad()
        img_embeds = image_encoder(images)
        txt_embeds = text_encoder(texts)
        loss = criterion(img_embeds, txt_embeds)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dim = 256
    batch_size = 32
    num_epochs = 10

    # Define image transformations (resize, convert to tensor, normalize)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])

    # Paths to your COCO images and annotations
    images_dir = "./train2017"  # Change to your images directory
    captions_file = "./captions_train2017.json"  # Change to your JSON file

    # Create COCO dataset
    dataset = COCODataset(images_dir, captions_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Initialize models
    image_encoder = SimpleImageEncoder(output_dim=output_dim).to(device)
    # Update vocab_size based on your dataset's vocabulary size
    vocab_size = len(dataset.vocab)
    text_encoder = SimpleTextEncoder(vocab_size=vocab_size, embed_dim=256, hidden_dim=256, output_dim=output_dim).to(device)

    optimizer = torch.optim.Adam(list(image_encoder.parameters()) + list(text_encoder.parameters()), lr=1e-3)
    criterion = ContrastiveLoss(temperature=0.07)

    for epoch in range(num_epochs):
        loss = train_epoch(image_encoder, text_encoder, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

if __name__ == "__main__":
    main()