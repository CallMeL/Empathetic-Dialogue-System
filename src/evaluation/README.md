
# Evaluation
The evaluation.ipynb notebook is designed to evaluate multiple models using different metrics. It aims to identify the best-performing model based on specific criteria

## Explanation of the Data
| Data   | Description                                                                                                                                                                                | Trained Model                              |
| :---------------------------- |:--------------------------|-----------------|
| 59k_eachconv_eot              | Under `no_additional_tag` folder. <br />Facebook dataset with `endOfText` inserted after  every 2 sentences.                                                                               | 3 modified models <br/> single_conversation |
| 59k_wholeconv_eot             | Under `no_additional_tag` folder. <br />Facebook dataset with `endOfText` inserted at  the end of the whole conversation.                                                                  | whole_conversation                         |
| 59k_eachconv_eot_with_context | Under `context_tag` folder.<br />Facebook dataset with `endOfText`  <br />After every 2 sentences, including context.                                                                      | single_conversation_withcontext            |
| 59k_eachconv_eot_with_emotion | Under `emotion_file` folder.>Facebook dataset with `endOfText`  <br />After every 2 sentences, including emotion.                                                                          | single_conversation_withemotion            |
| with_gpt_data                 | Under `with_gpt_data` folder.  <br /> Based on  the question in 59k_eachconv_eot, we generated  the answer from ChatGPT 4omini, therefore we have 118k pairs of conversation               | single_conversation_withGPTdata_bs256, single_conversation_withGPTdata_withoutemotion |

# Results
This is a table with the results of the best performing models for each of the evaluation metrics

List of evaluated models:
- withoutemotion_single
- withoutemotion_whole
- withemotion
- withcontext
- gpt_withoutemotion
- gpt_blocksize_256

**For the small dataset (only facebook data)**

| Evaluation Metrics | Model               |
|--------------------|---------------------|
| BLEU              | gpt_blocksize_256  |
| BertScore         | withemotion         |
| GLUE              | withemotion         |
| Perplexity        | withcontext         |

**For the big dataset (facebook + gpt data)**

| Evaluation Metrics | Model               |
|--------------------|---------------------|
| BLEU              | gpt_withoutemotion  |
| BertScore         | gpt_withoutemotion  |
| GLUE              | withoutemotion_single   |
| Perplexity        | gpt_withoutemotion     |


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
It quantifies the model's "surprise" 😮 when encountering new data. Calculates the probability of a sentence under a specific language model.

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

```
BERTScore
Reference 1: I feel very sad today



```

## GLUE
GLUE, also known as General Language Understanding Evaluation, is an evaluation benchmark designed to measure the performance of language understanding models in a range of natural language processing (NLP) tasks.

GLUE has different possible tasks. For the purpose of our project maybe this one is the most interesting: SST-2 (Sentiment Analysis). Stanford Sentiment Treebank 2. Sentiment classification (positive or negative) of sentences.

Compares model-generated sentiments against reference sentiments for a dataset. In our pipeline we used an **emotion classifier** that uses the model bhadresh-savani/distilbert-base-uncased-emotion from Hugging Face’s Transformers library. 

Supported Classes: Joy, Anger, Sadness, Fear, Surprise, Love, Neutral
```
GLUE

```



