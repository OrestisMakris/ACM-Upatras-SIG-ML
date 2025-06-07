# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import random
# import matplotlib.pyplot as plt
# from tqdm import tqdm  # For progress bar
# from sklearn.manifold import TSNE

# # Define hyperparameters
# VOCAB_SIZE = 1_000_000  # 3 million words
# EMBED_DIM = 100         # 100-dimensional embeddings
# NEGATIVE_SAMPLES = 10   # Number of negative samples per word
# BATCH_SIZE = 512        # Large batch for efficiency
# EPOCHS = 100            # 100 training epochs
# LEARNING_RATE = 0.025   # Standard learning rate

# # Use GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Create a synthetic dataset - Pairs of (center_word, context_word)
# def generate_fake_data(num_samples=6_000_000):
#     np.random.seed(42)
#     return [(random.randint(0, VOCAB_SIZE-1), random.randint(0, VOCAB_SIZE-1)) for _ in range(num_samples)]

# # Load dataset
# word_pairs = generate_fake_data()

# # Convert dataset into tensors
# word_pairs = torch.tensor(word_pairs, dtype=torch.long).to(device)

# # Define Skip-gram Model with Negative Sampling
# class Word2Vec(nn.Module):
#     def __init__(self, vocab_size, embed_dim):
#         super(Word2Vec, self).__init__()
#         self.input_embeddings = nn.Embedding(vocab_size, embed_dim, sparse=True)
#         self.output_embeddings = nn.Embedding(vocab_size, embed_dim, sparse=True)
#         self.init_emb()

#     def init_emb(self):
#         nn.init.uniform_(self.input_embeddings.weight, -0.5 / EMBED_DIM, 0.5 / EMBED_DIM)
#         nn.init.constant_(self.output_embeddings.weight, 0)

#     def forward(self, center, context, negatives):
#         # Positive sample similarity (dot product)
#         pos_score = torch.mul(self.input_embeddings(center), self.output_embeddings(context)).sum(dim=1)
#         pos_score = torch.sigmoid(pos_score)

#         # Negative sample similarity
#         neg_score = torch.mul(self.input_embeddings(center).unsqueeze(1), self.output_embeddings(negatives)).sum(dim=2)
#         neg_score = torch.sigmoid(-neg_score)

#         # Loss function: -log(sigmoid(pos)) - sum(log(sigmoid(-neg)))
#         loss = -torch.log(pos_score + 1e-9).mean() - torch.log(neg_score + 1e-9).sum(dim=1).mean()
#         return loss

# # Create Model
# model = Word2Vec(VOCAB_SIZE, EMBED_DIM).to(device)
# optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

# # Generate negative samples
# def get_negative_samples(batch_size, num_neg_samples):
#     return torch.randint(0, VOCAB_SIZE, (batch_size, num_neg_samples), device=device)

# # Store losses for visualization
# loss_history = []

# # Training Loop with TQDM
# for epoch in range(EPOCHS):
#     total_loss = 0
#     np.random.shuffle(word_pairs.cpu().numpy())  # Shuffle dataset
    
#     with tqdm(total=len(word_pairs) // BATCH_SIZE, desc=f"Epoch {epoch+1}/{EPOCHS}") as pbar:
#         for i in range(0, len(word_pairs), BATCH_SIZE):
#             batch = word_pairs[i:i+BATCH_SIZE]
#             if batch.shape[0] < BATCH_SIZE: break  # Ignore last small batch

#             # Extract center and context words
#             center_words, context_words = batch[:, 0], batch[:, 1]
#             negative_samples = get_negative_samples(BATCH_SIZE, NEGATIVE_SAMPLES)

#             # Forward pass
#             loss = model(center_words, context_words, negative_samples)

#             # Backpropagation
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()
#             pbar.update(1)

#     avg_loss = total_loss / (len(word_pairs) // BATCH_SIZE)
#     loss_history.append(avg_loss)
#     print(f"Epoch {epoch+1}/{EPOCHS} - Avg Loss: {avg_loss:.4f}")

# # Plot Loss
# plt.figure(figsize=(10,5))
# plt.plot(range(1, EPOCHS+1), loss_history, label="Training Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Word2Vec Training Loss Over Time")
# plt.legend()
# plt.show()

# print("Training complete! Word vectors are ready. 🚀")

# # ====== Embedding Space Visualization ======
# def plot_embeddings(embeddings, words, title="Word Embeddings Visualization"):
#     tsne = TSNE(n_components=2, perplexity=30, random_state=42)
#     reduced_embeddings = tsne.fit_transform(embeddings)

#     plt.figure(figsize=(12, 8))
#     plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], marker="o", alpha=0.7)

#     for i, word in enumerate(words):
#         plt.annotate(word, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=9, alpha=0.8)

#     plt.title(title)
#     plt.xlabel("TSNE Dimension 1")
#     plt.ylabel("TSNE Dimension 2")
#     plt.show()

# # Select a random subset of words for visualization (e.g., 300 words)
# NUM_WORDS_TO_VISUALIZE = 300
# random_indices = np.random.choice(VOCAB_SIZE, NUM_WORDS_TO_VISUALIZE, replace=False)

# # Get word embeddings for selected words
# word_embeddings = model.input_embeddings.weight[random_indices].detach().cpu().numpy()

# # Fake words (replace with real words if using a real dataset)
# word_labels = [f"word_{i}" for i in random_indices]

# # Plot the embeddings
# plot_embeddings(word_embeddings, word_labels)






import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bar
from sklearn.manifold import TSNE

# Import TorchText tools
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# ===== Hyperparameters =====
EMBED_DIM = 100         # 100-dimensional embeddings
NEGATIVE_SAMPLES = 5    # Number of negative samples per word
BATCH_SIZE = 512        # Batch size for training
EPOCHS = 10             # Number of training epochs
LEARNING_RATE = 0.025   # Learning rate
WINDOW_SIZE = 2         # Context window size (left and right)
MAX_VOCAB_SIZE = 10000  # Maximum vocabulary size

# ===== Device setup =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Load and Process the Dataset =====
# Load WikiText2 training data
train_iter = list(WikiText2(split='train'))

# Tokenizer (simple English tokenizer)
tokenizer = get_tokenizer("basic_english")

# Build vocabulary from the training data
def yield_tokens(data_iter):
    for item in data_iter:
        yield tokenizer(item)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"], max_tokens=MAX_VOCAB_SIZE)
vocab.set_default_index(vocab["<unk>"])
VOCAB_SIZE = len(vocab)
print("Vocabulary size:", VOCAB_SIZE)

# Generate training pairs using a sliding window
pairs = []
for line in train_iter:
    tokens = tokenizer(line)
    if not tokens:
        continue
    indices = [vocab[token] for token in tokens]
    for i, center in enumerate(indices):
        # For each center word, consider words in the window around it as context
        for j in range(max(0, i - WINDOW_SIZE), min(len(indices), i + WINDOW_SIZE + 1)):
            if i != j:
                pairs.append((center, indices[j]))
print("Total training pairs:", len(pairs))

# Optionally, limit the number of pairs for faster training (here, we use the first 100k pairs)
pairs = pairs[:100000]
print("Using", len(pairs), "pairs for training.")

# Convert pairs to a tensor
word_pairs = torch.tensor(pairs, dtype=torch.long).to(device)

# ===== Define the Word2Vec Model (Skip-gram with Negative Sampling) =====
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(Word2Vec, self).__init__()
        self.input_embeddings = nn.Embedding(vocab_size, embed_dim, sparse=True)
        self.output_embeddings = nn.Embedding(vocab_size, embed_dim, sparse=True)
        self.init_emb()

    def init_emb(self):
        nn.init.uniform_(self.input_embeddings.weight, -0.5 / EMBED_DIM, 0.5 / EMBED_DIM)
        nn.init.constant_(self.output_embeddings.weight, 0)

    def forward(self, center, context, negatives):
        # Compute positive score (dot product between center and context embeddings)
        pos_score = torch.mul(self.input_embeddings(center), self.output_embeddings(context)).sum(dim=1)
        pos_score = torch.sigmoid(pos_score)

        # Compute negative score (dot product for negative samples)
        neg_score = torch.mul(self.input_embeddings(center).unsqueeze(1), self.output_embeddings(negatives)).sum(dim=2)
        neg_score = torch.sigmoid(-neg_score)

        # Loss: -log(sigmoid(pos)) - sum(log(sigmoid(-neg)))
        loss = -torch.log(pos_score + 1e-9).mean() - torch.log(neg_score + 1e-9).sum(dim=1).mean()
        return loss

# Create model and optimizer
model = Word2Vec(VOCAB_SIZE, EMBED_DIM).to(device)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

# Function to get negative samples for a batch
def get_negative_samples(batch_size, num_neg_samples):
    return torch.randint(0, VOCAB_SIZE, (batch_size, num_neg_samples), device=device)

# ===== Training Loop =====
loss_history = []
num_batches = len(word_pairs) // BATCH_SIZE

for epoch in range(EPOCHS):
    total_loss = 0
    # Shuffle the pairs each epoch
    indices = np.random.permutation(len(word_pairs))
    word_pairs = word_pairs[indices]
    
    with tqdm(total=num_batches, desc=f"Epoch {epoch+1}/{EPOCHS}") as pbar:
        for i in range(0, len(word_pairs), BATCH_SIZE):
            batch = word_pairs[i:i+BATCH_SIZE]
            if batch.shape[0] < BATCH_SIZE:
                break  # Skip the last small batch

            center_words, context_words = batch[:, 0], batch[:, 1]
            negative_samples = get_negative_samples(BATCH_SIZE, NEGATIVE_SAMPLES)

            loss = model(center_words, context_words, negative_samples)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.update(1)
            
    avg_loss = total_loss / num_batches
    loss_history.append(avg_loss)
    print(f"Epoch {epoch+1}/{EPOCHS} - Avg Loss: {avg_loss:.4f}")

# Plot the training loss over epochs
plt.figure(figsize=(10,5))
plt.plot(range(1, EPOCHS+1), loss_history, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Word2Vec Training Loss Over Time")
plt.legend()
plt.show()

print("Training complete! Word vectors are ready.")

# ===== Embedding Space Visualization with t-SNE =====
def plot_embeddings(embeddings, words, title="Word Embeddings Visualization"):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)
    plt.figure(figsize=(12, 8))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], marker="o", alpha=0.7)
    for i, word in enumerate(words):
        plt.annotate(word, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]), fontsize=9, alpha=0.8)
    plt.title(title)
    plt.xlabel("TSNE Dimension 1")
    plt.ylabel("TSNE Dimension 2")
    plt.show()

# Select a random subset of words to visualize (e.g., 50 words)
NUM_WORDS_TO_VISUALIZE = 50
random_indices = np.random.choice(VOCAB_SIZE, NUM_WORDS_TO_VISUALIZE, replace=False)
word_embeddings = model.input_embeddings.weight[random_indices].detach().cpu().numpy()

# Get corresponding word labels (if your vocab supports it)
if hasattr(vocab, "get_itos"):
    itos = vocab.get_itos()
    word_labels = [itos[i] for i in random_indices]
else:
    word_labels = [f"word_{i}" for i in random_indices]

plot_embeddings(word_embeddings, word_labels)
