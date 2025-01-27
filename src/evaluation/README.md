
# Evaluation
The evaluation.ipynb notebook is designed to evaluate multiple models using different metrics. It aims to identify the best-performing model based on specific criteria

## Explanation of the Data
We use 10% of the gpt-enhanced training data (10% of the 60k)

## Results

List of evaluated models:
- single_conversation
- single_conversation_rope
- single_conversation_relative
- whole_conversation
- single_conversation_withemotion
- single_conversation_withcontext
- single_conversation_withGPTdata_withoutemotion
- single_conversation_withGPTdata_bs256,

**For the big dataset (facebook + gpt data)**

| Model                             | BLEU-1               | BLEU-2               | BLEU-3               | BLEU-4               | Bert F1            | GLUE                  | Perplexity          |
|-----------------------------------|----------------------|----------------------|----------------------|----------------------|--------------------|-----------------------|---------------------|
| single_conversation               | 0.00587468362062949  | 0.010446828923058333 | 0.007117341692084967 | 0.00587468362062949  | 0.8568010926246643 | **0.4727166374195705** | 84066.95248548205   |
| single_conversation_rope          | 0.005886990900708465 | ""                   | ""                   | ""                   | 0.48094046115875244 | 0.32974011865964736   | 3776063.144329027   |
| single_conversation_relative      | 0.005992347214885741 | ""                   | ""                   | ""                   | 0.36667502297986126 | 0.32974011865964736   | inf                 |
| whole_conversation                | 0.005539267542303131 | 0.009850365417174566 | 0.006710975836043849 | 0.005539267542303131 | 0.8480852246284485 | 0.43603242249519514   | 29549.33700155131   |
| single_conversation_withemotion   | 0.005846493825725561 | 0.010396699591207457 | 0.007083188975867789 | 0.005846493825725561 | 0.8573451042175293 | 0.4721316954959472    | 2008198.4398578384  |
| single_conversation_withcontext   | 0.005802720584588993 | 0.010318858537783629 | 0.007030156483523018 | 0.005802720584588993 | 0.8359043002128601 | 0.45884515751650373   | 63035.03316265359   |
| single_conversation_withGPTdata_withoutemotion | **0.006432147502200551** | **0.011438155465496527** | **0.007792724603294758** | **0.006432147502200551** | **0.8576485514640808** | 0.47171387983621627 | **28233.374812001217** |
| single_conversation_withGPTdata_bs256 | 0.0062091122888948  | 0.011041536537961272 | 0.007522511273526714 | 0.0062091122888948  | 0.48648038506507874 | 0.34519929806969163   | 218114.86185134976  |


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


