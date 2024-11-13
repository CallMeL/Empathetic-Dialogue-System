from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch, torchvision
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def define_data():
    # Example Data
    reference_sentences = [
        "I feel very sad today",
        "He is extremely happy with his results",
        "She found the book boring and uninspiring",
        "The weather is beautiful and sunny",
        "I need help with my homework"
    ]

    model_outputs = [
        "I feel very happy today",          # Hypothetical bad output
        "He is extremely happy with results",  # Missing "his"
        "She found the book boring",        # Missing some words
        "The weather is beautiful and sunny", # Exact match
        "I need some help with homework"    # Slight variation
    ]
    return reference_sentences, model_outputs

# *********************************************************************************************************************
# BLEU Score Calculation
# *********************************************************************************************************************
def calculate_bleu(reference_sentences, model_outputs):

    # Ensure that the references and outputs have the same length
    assert len(reference_sentences) == len(model_outputs), "Mismatch in reference and output count."

    bleu_scores = []
    smoothing_function = SmoothingFunction().method1  # To avoid 0 scores due to short sentences

    for ref, output in zip(reference_sentences, model_outputs):
        # Tokenize each sentence (split by words)
        reference_tokens = [ref.split()]  # BLEU expects a list of lists for references
        output_tokens = output.split()
        
        # Calculate BLEU score
        bleu = sentence_bleu(reference_tokens, output_tokens, smoothing_function=smoothing_function)
        bleu_scores.append(bleu)

        print(f"Reference: {ref}")

    return bleu_scores

def print_bleu_scores(reference_sentences, model_outputs):
    bleu_scores = calculate_bleu(reference_sentences, model_outputs)
    for i, score in enumerate(bleu_scores):
            print(f"Reference {i+1}: {reference_sentences[i]}")
            print(f"Model Output {i+1}: {model_outputs[i]}")
            print(f"BLEU Score: {score:.2f}\n")

# *********************************************************************************************************************
# Perprexity Score Calculation
# *********************************************************************************************************************

def calculate_perplexity(sentences, model_name='gpt2'):
    print(f"Calculating perplexity using model: {model_name}\n")
    """
    Calculate the perplexity of a list of sentences using a specified pre-trained language model.

    Args:
        sentences (list of str): The sentences to evaluate.
        model_name (str): The name of the pre-trained model to use.

    Returns:
        list of float: Perplexity scores for each sentence.
    """
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    # Load pre-trained model (weights)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    # Determine the device: MPS if available, else CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device for computation.")
    else:
        device = torch.device("cpu")
        print("MPS not available. Falling back to CPU.")

    # Move the model to the selected device
    model.to(device)

    perplexities = []

    with torch.no_grad():
        for sentence in sentences:
            # Encode the sentence
            inputs = tokenizer(sentence, return_tensors='pt')
            input_ids = inputs['input_ids'].to(device)

            # Get the loss (cross-entropy) for the sentence
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            # Calculate perplexity
            perplexity = torch.exp(loss).item()
            perplexities.append(perplexity)

            print(f"Sentence: \"{sentence}\"")
            print(f"Perplexity: {perplexity:.2f}\n")

    return perplexities

def print_perplexity_scores(reference_sentences, model_outputs):
    perplexity_scores = calculate_perplexity(model_outputs)
    for i, score in enumerate(perplexity_scores):
        print(f"Reference {i+1}: {reference_sentences[i]}")
        print(f"Model Output {i+1}: {model_outputs[i]}")
        print(f"Perplexity: {score:.2f}\n")



if __name__ == "__main__":
    reference_sentences, model_outputs = define_data()
   # print_bleu_scores(reference_sentences, model_outputs)

    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    print_perplexity_scores(reference_sentences, model_outputs)