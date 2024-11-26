# Weekly meeting 5
Date: 26.11.2024
## What we discussed

### inside the group
1. try to encode the emotional tag
2. try to add more data (daily dialog and extreme emotional case)
3. 1 and 3 are big enough, no 3rd one
### with the teachers
1. We are not planning to train another brand new model. does it count, that we slightly change the nanoGPT decoder: adding an embeding layer for the emotional tag.
2. right now the output looks like it does not understand the input at all,  Question: Should we separate the training input and label.  (!!!!!!!!!)
3. How many training iteration would be enough, is early stopping make sense for the LM? now we tried 28k iter, it has the problem we mentioned in 1
4. do we add somthing like "sorry I don't understand, could you explain it again" to avoid our model output random stuff when it doesn't understand.
## Goals for next week

## Check the todos from last meeting