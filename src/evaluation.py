from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


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

# Call the function
bleu_scores = calculate_bleu(reference_sentences, model_outputs)

# Display the results
for i, score in enumerate(bleu_scores):
    print(f"Reference {i+1}: {reference_sentences[i]}")
    print(f"Model Output {i+1}: {model_outputs[i]}")
    print(f"BLEU Score: {score:.2f}\n")