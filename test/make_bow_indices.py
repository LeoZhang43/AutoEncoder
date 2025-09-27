import json
import re
import argparse

#python make_bow_indices.py papers.json vocabulary.json papers_with_bow.json

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Keep only alphabets and spaces
    text = re.sub(r'[^a-z\s]', '', text)
    # Split into words
    words = text.split()
    # Remove very short words
    words = [w for w in words if len(w) > 1]
    return words

def main(papers_file, vocab_file, output_file):
    # Load papers
    with open(papers_file, "r") as f:
        papers = json.load(f)

    # Load vocabulary mapping
    with open(vocab_file, "r") as f:
        vocab_data = json.load(f)
        vocab_to_idx = vocab_data["vocab_to_idx"]

    papers_with_bow = []
    for paper in papers:
        abstract = paper["abstract"]
        tokens = clean_text(abstract)

        # Convert tokens to indices (skip OOV words)
        bow_indices = sorted(set(vocab_to_idx.get(w, 0) for w in tokens if w in vocab_to_idx))

        papers_with_bow.append({
            "arxiv_id": paper["arxiv_id"],
            "bow_indices": bow_indices,
            "title": paper["title"],
            "authors": paper["authors"],
            "categories": paper["categories"]
        })

    # Save new file
    with open(output_file, "w") as f:
        json.dump(papers_with_bow, f, indent=2)

    print(f"✅ Saved {len(papers_with_bow)} papers with BoW indices to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("papers_file", help="Input papers.json")
    parser.add_argument("vocab_file", help="Vocabulary JSON (from training)")
    parser.add_argument("output_file", help="Output file with bow indices")
    args = parser.parse_args()

    main(args.papers_file, args.vocab_file, args.output_file)
