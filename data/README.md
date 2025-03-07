# Dataset
_Check ```EDA/69kdataset.ipynb``` for better understanding_

## Source of data
`emotion-emotion_69k.csv` is downloded from the [Empathetic Dialogues (Facebook AI) 25k](https://www.kaggle.com/datasets/atharvjairath/empathetic-dialogues-facebook-ai/data). There are about 50 misplaced data, we just deleted the wrong raws.

## Data structure
+ The number of data: 64,594
+ We have 4 columns
   + **Situation**: Describes the theme or concept of the conversation.
   + **Emotion**: Categorizes the sentiment of the conversation.
   + **Empathetic_dialogues**: Represents the input data, which serves as the conversation opener based on the given situation.
   + **Labels**: Provides the empathetic response to the input data, functioning as the desired output.

        |  | Situation | emotion | empathetic_dialogues | labels |
        |---|:---:|:---:|:---:|:---:|
        | 0 | I remember going to the fireworks with my best friend. There was a lot of people, but it only felt like us in the world. | sentimental | Customer :I remember going to see the fireworks with my best friend. It was the first time we ever spent time alone together. Although there was a lot of people, we felt like the only people in the world.\nAgent  | Was this a friend you were in love with, or just a best friend? |
        | 64635 | I woke up this morning to my wife telling me she was pregnant! | surprised | Customer :Oh hey that's awesome! That is awesome right?\nAgent : | It is soooo awesome. We have been wanting a baby for so long. I can't wait, but I was shocked out of a dead sleep! |

## Data preprocessing
**1. Reduce Categories of emotion from 30 to 8**
   + Since the classification of the dataset we have is too detailed with 30 categories, we have grouped it into 8 larger categories
   + Result: ```disappointed, grateful, annoyed, disgusted, impressed, prepared, afraid, excited```
  
**2. Remove Prefix and Suffix from empathetic_dialogues**
   + Since all the sentences in 'empathetic_dialogues' start with 'Customer :' and end with '\nAgent :', we have removed these strings.
  
**3. Remove Improper Elements and Handling Missing Values**
   + There are improper elements such as those containing only numbers, so filter them out if they are less than **3 characters** long, and also remove any missing values.
   + Now we have **59,835** data

**_4. Handle repeated conversations_**
+ We noticed that the data was repeating itself, with the customer and agent switching each other
  + e.g. In this table, you can see that the output (labels) from Conv1 and the input (empathetic dialogues) from Conv2 are reused to form a new conversation pair in the Repeated conversation.

      |   | empathetic_dialogues (desired input)                                                                                     | labels (desired output)                                      |
      |---|--------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------|
      | Conv 1  | I remember going to see the fireworks with my best friend. It was the first time we ever spent time alone together. Although there were many people, we felt like the only people in the world. | **Was this a friend you were in love with, or just a best friend?** |
      | Conv 2  | **This was a best friend. I miss her.**                                                                                   | Where has she gone?                                         |
      | Repeated Conv | Was this a friend you were in love with, or just a best friend?                                                           | This was a best friend. I miss her.                         |
+ We chose to treat them as a form of data augmentation, **deciding to retain them**, as they might contribute to a more nuanced understanding of dialogue by the model. The total number of such instances was found to be 40,779.

**_5. Remove stopwords_**
+ We applied a stopword removal process to the dataset for natural language processing.
+ However, this processed data was not used in the model training, as the model needed to learn the natural structure of sentences.
+ Instead, it was utilized later during the data analysis phase, specifically for word analysis within our dataset

| Situation                                                                                                  | Emotion    | Empathetic_dialogues                                                    | Labels                          | Cleaned_input                | Cleaned_output            |
|------------------------------------------------------------------------------------------------------------|------------|-------------------------------------------------------------------------|---------------------------------|------------------------------|---------------------------|
| I remember going to the fireworks with my best friend. There was a lot of people, but it only felt like us in the world. | sentimental | Was this a friend you were in love with, or just a best friend?          | This was a best friend. I miss her. | friend love best friend       | best friend miss          |

## Final Results of the data
|  | **empathetic_dialogues** | **labels** |
|---|:---:|:---:|
| 0 | I remember going to see the fireworks with my best friend. It was the first time we ever spent time alone together. Although there was a lot of people, we felt like the only people in the world. | Was this a friend you were in love with, or just a best friend? |
| 1 | This was a best friend. I miss her.| Where has she gone? |
| 2 | We no longer talk. | Oh was this something that happened because of an argument? |
| 3 | Was this a friend you were in love with, or just a best friend? | This was a best friend. I miss her. |
| 4 | Where has she gone? | We no longer talk. |

## Exploratory Data Analysis (EDA)
### Sentence-Level Analysis
**1. Sentence Length Analysis**
   + We analysed the length of each input (empathetic_dialogues) and output (labels) sentence
   + Average and Median Sentence Length by Character Count
        |  | empathetic_dialogues | labels |
        |---|:---:|:---:|
        | Mean | 68.00 | 62.68 |
        | Median | 59.0 | 54 |
   + Examples of sentences with lengths close to the average are as follows:
     + Input (empathetic_dialogues) 
        + Got rejected from a place I wanted to work, not once but three times
        + I just really wanted some ice cream! Now I know their hours, though.
        + I do sales work, but he always lies to us and takes our bonus money.
        + Yeah, thank you! at which situation did you feel hope for your life?
        + I hear ya.. I hope you find one soon... wishing you all of the best!
      + Output (labels)
        + Was this a friend you were in love with, or just a best friend?
        + The grass makes me itchy, But the shower afterward feels great.
        + I still took it since it was late but I rode in the front seat.
        + Oh no... were they relaxed about it or did it cause a problem? 
        + That's really considerate of you. Do they need your help a lot?

**2. Sentiment Analysis**  
- Utilized the `all-MiniLM-L6-v2` model from `SentenceTransformer` to convert each sentence into a vector representation and measured the similarity between these vectors using cosine similarity.
- Values closer to -1 indicate negative sentiment, 0 represents neutral, and values closer to 1 indicate positive sentiment.  
- Results
  - For the input dataset (empathetic_dialogues), the majority of similarity scores ranged between 0 and 0.3, indicating that most sentences exhibited weak similarity. Additionally, similarity scores exceeding 0.5 were extremely rare, suggesting that highly related sentences were relatively uncommon. 
  - For the output (model's response), 


### Word-Level Analysis
**1. Vocabulary Size** 
+ We calculated the vocabulary size for both the input and output sentences after removing stop words.
+ Results
  + Input (empathetic_dialogues)
    + The vocabulary size: 18,273
    + The most frequently occurring 5 words:  "im", "really", "thats", "get", "like"
  + Output (labels)
    + The vocabulary size: 17,744
    + The most frequently occurring 5 words: "thats", "good", "im", "oh", and "like"

**2. LDA(Latent Dirichlet Allocation)** 
- 분석
- Wordcloud



____

## The training data format
We generate txt files from the csv file in `data2txt.ipynb`, and then generate 
`train.
bin` 
and 
`val.bin` for the model training with `prepare.py`
  
Inside the txt file, we choose to use tags like `<context> ` `<bot>`  `<human>` 
and 
`<endOfText>` 
based on [this blog](https://vatsadev.hashnode.dev/making-nanochatgpt-nanogpt-chat-oriented)

| Data   | Description                                                                                                                                                                                | Trained Model                                                                         |
| :---------------------------- |:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| 59k_eachconv_eot              | Under `no_additional_tag` folder. <br />Facebook dataset with `endOfText` inserted after  every 2 sentences.                                                                               |single_conversation, <br/> single_conversation_rope,  <br/>single_conversation_relative|
| 59k_wholeconv_eot             | Under `no_additional_tag` folder. <br />Facebook dataset with `endOfText` inserted at  the end of the whole conversation.                                                                  | whole_conversation                                                                    |
| 59k_eachconv_eot_with_context | Under `context_tag` folder.<br />Facebook dataset with `endOfText`  <br />After every 2 sentences, including context.                                                                      | single_conversation_withcontext                                                       |
| 59k_eachconv_eot_with_emotion | Under `emotion_file` folder.>Facebook dataset with `endOfText`  <br />After every 2 sentences, including emotion.                                                                          | single_conversation_withemotion                                                       |
| with_gpt_data                 | Under `with_gpt_data` folder.  <br /> Based on  the question in 59k_eachconv_eot, we generated  the answer from ChatGPT 4omini, therefore we have 118k pairs of conversation               | single_conversation_withGPTdata_bs256, single_conversation_withGPTdata |