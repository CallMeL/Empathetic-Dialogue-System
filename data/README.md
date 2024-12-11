# Dataset
_Check ```src/69kdataset.ipynb``` for better understanding_

## Source of data
1. `emotion-emotion_69k.csv` is downloded from [https://www.kaggle.com/datasets/atharvjairath/empathetic-dialogues-facebook-ai/data](the Empathetic Dialogues (Facebook AI) 25k). There are about 50 misplaced data, we just deleted the wrong raws.

## the txt data for training
TODO: Add explaination here
https://vatsadev.hashnode.dev/making-nanochatgpt-nanogpt-chat-oriented
RedPajama-3b model

## Data structure
**1. The number of data: 64,636**

**2. We have 4 columns**
   + **Situation**: described the theme/concept of conversation
   + **Emotion**: Categorised the sentiment of the conversation
   + **Empathetic_dialogues**: customer's input, so worked as an input data
   + **Labels**: empathetic response of input data, sp worked an an desirable ouput

        |  | Situation | emotion | empathetic_dialogues | labels |
        |---|:---:|:---:|:---:|:---:|
        | 0 | I remember going to the fireworks with my best friend. There was a lot of people, but it only felt like us in the world. | sentimental | Customer :I remember going to see the fireworks with my best friend. It was the first time we ever spent time alone together. Although there was a lot of people, we felt like the only people in the world.\nAgent  | Was this a friend you were in love with, or just a best friend? |
        | 64635 | I woke up this morning to my wife telling me she was pregnant! | surprised | Customer :Oh hey that's awesome! That is awesome right?\nAgent : | It is soooo awesome. We have been wanting a baby for so long. I can't wait, but I was shocked out of a dead sleep! |

## Data Engineering and EDA
**1. Reducing Categories of emotion from 30 to 8**
   + Since the classification of the dataset we have is too detailed with 30 categories, we have grouped it into 8 larger categories
   + Result: ```disappointed, grateful, annoyed, disgusted, impressed, prepared, afraid, excited```
  
**2. Removing Prefix and Suffix from empathetic_dialogues**
   + Since all the sentences in 'empathetic_dialogues' start with 'Customer :' and end with '\nAgent :', I have removed these strings.
  
**3. Removing Improper Elements and Handling Missing Values**
   + There are improper elements such as those containing only numbers, so filter them out if they are less than 3 **characters** long, and also remove any missing values.
   + Now we have **59,835** data
  
**4. Length Analysis**
   + analysed the length of each input (empathetic_dialogues) and output (labels) sentence
   + Average and Median Sentence Length by Character Count
        |  | empathetic_dialogues | labels |
        |---|:---:|:---:|
        | Mean | 68.28 | 62.85 |
        | Median | 59.0 | 54 |
  - [ ] need to consider whether to use all the sentences or select some based on sentence length
  
**5. Delete repeated sentencess**
   + I noticed that the data was repeating itself, with the customer and agent switching each other
   + e.g. In this table, you can see that the input (empathetic dialogues) in rows 3 and 4 are exactly the same as the output (labels) in rows 0 and 1, and the outputs of these are exactly the same as the inputs in rows 1 and 2.
   + WE DECIDED NOT DELETING THEM
  
|  | **empathetic_dialogues** | **labels** |
|---|:---:|:---:|
| 0 | I remember going to see the fireworks with my best friend. It was the first time we ever spent time alone together. Although there was a lot of people, we felt like the only people in the world. | Was this a friend you were in love with, or just a best friend? |
| 1 | This was a best friend. I miss her.| Where has she gone? |
| 2 | We no longer talk. | Oh was this something that happened because of an argument? |
| 3 | Was this a friend you were in love with, or just a best friend? | This was a best friend. I miss her. |
| 4 | Where has she gone? | We no longer talk. |

**6. Preprocessed the data**
+ removed stopwords, etc.

**prepared a `.txt` file for input.**
   + However, apparently nanoGPT does not require removing stopwords and so on
  
**7. Data analysis (input and output)**
   - **A. LDA for word clustering**  
     - Conducted word clustering using LDA.  
     - However, the words were not grouped based on emotions.  
     - Further thought is needed on how to interpret this result.

   - **B. Sentiment Analysis**  
     - Values closer to -1 indicate negative sentiment, 0 represents neutral, and values closer to 1 indicate positive sentiment.  
     - For input (user's utterance), it would be beneficial for the data to have more values closer to -1 or 1 for model training. Unfortunately, most of the data is clustered around 0. Therefore, generating more emotionally charged utterances using LLMs like ChatGPT might be a good approach.  
     - For output (model's response), the data is primarily distributed between 0 and 1, which seems desirable.  
     - Since we have sentiment data in our dataset, we plan to verify the accuracy of this analysis by comparing them.

**8. make txt files: ```data2txt.ipynb```**
+ **59k_eachconv_eot**: full data (59k) & have endOfText in every 2 sentences
+ **59k_wholeconv_eot**: full data (59k) & have endOfText at the very end of the whole conversation within the same context
+ **38k_eachconv_eot**: 38k data (after deleting repeated conv) & have endOfText in every 2 sentences
+ **38k_wholeconv_eot**: data (after deleting repeated conv) & have endOfText at the very end of the whole conversation within the same context


## How to get  `train.bin` and `val.bin`
### 🔊 LATEST UPDATE 🔊
With our new dataset,

**9. Sentence Length**


**10.Vocabulary size**


**11. similarity analysis**


```
cd data
python ./prepare.py
```