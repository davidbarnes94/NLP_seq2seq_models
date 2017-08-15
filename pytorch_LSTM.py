import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

###LOAD THE DATA
inputs = [autograd.Variable(torch.randn(1, 3)) for _ in range(5)]


###DEFINE THE NEURAL NETWORK
lstm = nn.LSTM(3, 3) #both input and output have dimension 3

##Initialize the hidden state
hidden = (autograd.Variable(torch.randn(1, 1, 3)), autograd.Variable(torch.randn(1, 1, 3)))

for i in inputs:
    out, hidden = lstm(i.view(1, 1, -1), hidden)
    #view() reshapes a tensor

inputs = torch.cat(inputs).view(len(inputs), 1, -1)
#cat() concatenates a sequence
hidden = (autograd.Variable(torch.randn(1, 1, 3)), autograd.Variable(torch.randn(
    1, 1, 3
)))
out, hidden = lstm(inputs, hidden)
#print(out)
print("=================================")
#print(hidden)


training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Every dog read that book".split(), ["NN", "V", "DET", "NN"])
]
# word_to_ix = {}
# for sent, tags in training_data:
#     for word in sent:
#         if word not in word_to_ix:
#             word_to_ix[word] = len(word_to_ix)
#         print(word_to_ix)
#

word_to_ix = {"hello": 0}
embeds = nn.Embedding(1, 5)  # 2 words in vocab, 5 dimensional embeddings
lookup_tensor = torch.LongTensor([word_to_ix["hello"]])
hello_embed = embeds(autograd.Variable(lookup_tensor))
#print(hello_embed)

###############################################################################################
def prepare_sequence(seq, to_ix):
    """

    :param seq: the list of words in a sentence
    :param to_ix: word_to_ix
    :return: tensor of all the indices of the words
    """
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
#create word dictionary
word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        #The hidden layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        #output layer
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence) #embed in 6 dim each index corresponding to a word in the sentence
        print("embeds: {0}".format(embeds)) # 5x6
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        print("lstm_out: {0}".format(lstm_out)) #output from the hidden layer # 5 x 1 x 6
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1)) #give value of tag for each word in sentence
        print("tag_space: {0}".format(tag_space)) # 5x3
        tag_scores = F.log_softmax(tag_space)
        print("tag_scores: {0}".format(tag_scores)) # 5x3
        return tag_scores

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
inputs = prepare_sequence(training_data[0][0], word_to_ix)
#print(model)

#the forward method does this
tag_scores = model(inputs)
print(tag_scores)

for epoch in range(3):  # again, normally you would NOT do 300 epochs, it is toy data
    print("==============EPOCH {0} ===============".format(epoch))
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Variables of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

# See what the scores are after training
print("AFTER TRAINING")
inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)
# The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
#  for word i. The predicted tag is the maximum scoring tag.
# Here, we can see the predicted sequence below is 0 1 2 0 1
# since 0 is index of the maximum value of row 1,
# 1 is the index of maximum value of row 2, etc.
# Which is DET NOUN VERB DET NOUN, the correct sequence!
print(tag_scores)

