# Weekly meeting 5
Date: 26.11.2024
## What we discussed

### inside the group
1. try to encode the emotional tag
2. try to add more data (daily dialog and extreme emotional case)
3. 1 and 3 are big enough, no 3rd one
⚠️ After the discussion with the teacher, we will not do the changing model things for now.
### with the teachers
1. We are not planning to train another brand new model. does it count, that we slightly change the nanoGPT decoder: adding an embeding layer for the emotional tag.
   1. Ansewer: don't do it now
2. right now the output looks like it does not understand the input at all,  Question: Should we separate the training input and label.  (!!!!!!!!!)
   1. Ansewer: currently our model can generate readable text, but it is not responsive to the input. It could be the problem of '<endOfText>' token actually tells the model to start a new conversation. 
   The issue will try to solve it by using the new data and train it all over again.
   The new data should only have '<endOfText>'  at the every of every conversation. For example
    ```
    <bot> I remember going to see the fireworks with my best friend. It was the first time we ever spent time alone together. Although there was a lot of people, we felt like the only people in the world. 
    <human> Was this a friend you were in love with, or just a best friend? 
    <bot> This was a best friend. I miss her. 
    <human> Where has she gone? 
    <bot> We no longer talk. 
    <human> Oh was this something that happened because of an argument? <endOfText>
    ```
3. How many training iteration would be enough, is early stopping make sense for the LM? now we tried 28k iter, it has the problem we mentioned in 1
4. do we add somthing like "sorry I don't understand, could you explain it again" to avoid our model output random stuff when it doesn't understand.
   1. Answer: yes we can
## Goals for next week
1. (J) New modified data (`<endOfText>` at the end of total conversation, see above), and replace it to `data/emotion/robot_human_tag/emotion.txt`
2. (J) Midterm talk 
3. (S) Data argument from ChatGPT API 
4. (Y) Train the model again with new modifier data and use validation matrix
## Check the todos from last meeting