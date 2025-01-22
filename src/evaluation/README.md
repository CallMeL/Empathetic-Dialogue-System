
# Evaluation
The evaluation.ipynb notebook is designed to evaluate multiple models using different metrics. It aims to identify the best-performing model based on specific criteria

## Explanation of the Data
We use 10% of the gpt-enhanced training data (10% of the 60k)

## Results

List of evaluated models:

1. single_conversation_withGPTdata_bs256
2. single_conversation_withGPTdata
3. single_conversation_withcontext
4. single_conversation_withemotion
5. single_conversation_rope
6. single_conversation_relative
7. single_conversation
8. whole_conversation

Table with the results of the best performing models for each of the evaluation metrics

| Model                                | BLEU    | Bert (F1) | GLUE     | Perplexity   |
|--------------------------------------|---------|-----------|----------|--------------|
| single_conversation_withGPTdata_bs256| 0.006209| 0.486480  | 0.345199 | 218114.8618  |
| single_conversation_withGPTdata      | **0.006432** | **0.857648**  | 0.471738 | 28233.3748   |
| single_conversation_withcontext      | 0.005803| 0.835904  | 0.458845 | 63035.0331   |
| single_conversation_withemotion      | 0.005846| 0.857451  | **0.4721316** | 2008198.439  |
| single_conversation_rope             |         |           |          |              |
| single_conversation_relative         |          |          |          |           |
| single_conversation                  | 0.005874| 0.8568010 | 0.4727166| 84066.9524   |
| whole_conversation                   | 0.005539| 0.848085  | 0.4360324| 29549.33700  |


# Explanation
## AVG BLEU: average of BLEU-1, -2, -3, -4

BLEU : Bilingual Evaluation Understudy focuses on n-gram overlap which works well for machine translation tasks but poorly captures the empathy of a response

It does not account for long-range dependencies (sequence context)

Doesn't evaluate if the sequence as a whole makes sense and if it is emotionally aligned with the input

BLEU does not require a language model because directly compares n-grams (word sequences) between the generated output and reference sentences to assess similarity. It doesn’t rely on the probability distribution of words but rather on the overlap of word sequences.


**Example:**
```
Reference 1: I feel very sad today
Model Output: I feel very happy today
BLEU Score: 0.29
```

## Perplexity 
It quantifies the model's "surprise" when encountering new data. Calculates the probability of a sentence under a specific language model.

Values range from 0-1. Lower surprise indicates better prediction accuracy.

Model-Dependent: Different language models may assign different probabilities to the same sentence, resulting in different perplexity scores.

Problem: A model might display low perplexity while maintaining a high error rate, indicating overconfidence in incorrect predictions

**Example**
```
Reference 1: I feel very sad today
Model Output 1: I feel very happy today
BERTScore Precisiion: 0.9777
BERTScore Recall: 0.9777
BERTScore F1: 0.977
```

## BERTScore
Another evaluation metrics possibility
Uses contextual embeddings from BERT to measure the semantic similarity between the generated response and the reference.

More robust for open-domain tasks and captures semantic and contextual alignment.

BERTScore compares outputs like BLEU, but it uses a pre-trained model’s embeddings (e.g. BERT or RoBERTa) to perform the comparison. It does not evaluate the model’s performance internally (as perplexity does).


## GLUE
GLUE, also known as General Language Understanding Evaluation, is an evaluation benchmark designed to measure the performance of language understanding models in a range of natural language processing (NLP) tasks.

GLUE has different possible tasks. For the purpose of our project maybe this one is the most interesting: SST-2 (Sentiment Analysis). Stanford Sentiment Treebank 2. Sentiment classification (positive or negative) of sentences.

Compares model-generated sentiments against reference sentiments for a dataset. In our pipeline we used an **emotion classifier** that uses the model bhadresh-savani/distilbert-base-uncased-emotion from Hugging Face’s Transformers library. 

Supported Classes: Joy, Anger, Sadness, Fear, Surprise, Love, Neutral


