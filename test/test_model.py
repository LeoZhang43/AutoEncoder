import torch
import torch.nn as nn
import json
import argparse

# python test_model.py model.pth papers_with_bow.json test_embeddings.json
class TextAutoencoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, embedding_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        embedding = self.encoder(x)
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding


# -----------------------------
# Testing function
# -----------------------------
def test_model(model_file, papers_file, output_file):
    # Load model + vocab
    checkpoint = torch.load(model_file, map_location="cpu")
    vocab_to_idx = checkpoint["vocab_to_idx"]
    config = checkpoint["model_config"]

    vocab_size = config["vocab_size"]
    hidden_dim = config["hidden_dim"]
    embedding_dim = config["embedding_dim"]

    # Rebuild model
    model = TextAutoencoder(vocab_size, hidden_dim, embedding_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load test papers with bow_indices
    with open(papers_file, "r") as f:
        papers = json.load(f)

    results = []
    loss_fn = nn.BCELoss()

    for paper in papers:
        bow_indices = paper["bow_indices"]

        # Create BoW vector
        bow_vec = torch.zeros(vocab_size)
        for idx in bow_indices:
            if idx < vocab_size:
                bow_vec[idx] = 1.0

        # Forward pass
        with torch.no_grad():
            reconstruction, embedding = model(bow_vec.unsqueeze(0))  # batch=1
            loss = loss_fn(reconstruction, bow_vec.unsqueeze(0))

        results.append({
            "arxiv_id": paper["arxiv_id"],
            "embedding": embedding.squeeze().tolist(),
            "reconstruction_loss": loss.item()
        })

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Saved embeddings for {len(results)} papers to {output_file}")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_file", help="Path to model.pth")
    parser.add_argument("papers_file", help="Path to papers_with_bow.json")
    parser.add_argument("output_file", help="Where to save embeddings.json")
    args = parser.parse_args()

    test_model(args.model_file, args.papers_file, args.output_file)
