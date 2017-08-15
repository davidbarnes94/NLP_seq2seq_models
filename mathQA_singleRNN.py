import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

#Training data with questions and the correct answers
trainingData = [('Add 3 and 5', 8), ('Multiply 9 and 2', 18), ('Divide 9 by 3', 3),
                ('John had 3 mangoes then Mary gave him 4 more. How much does he have now?', 7),
                ('Sum 50 and 5', 55), ('Adam went to the store with 10 dollars then bought an apple for 6 dollars.'
                                       'How much does he know have?', 4), ('Subtract 16 from 30', 14), ('Multiply 2 and 30', 60),
                ('Add 25 and 39', 64)]

testData = [('If Alex had 50 dollars in his account before he deposited 30 dollars. How much does he now have?', 80),
            ('Add 9 and 3', 12), ('Subtract 20 from 64', 34), ('Divide 360 by 4', 90), ('Multiply 12 and 3', 36),
            ('What is 2 by 2 by 2?', 8)]


def createWordDictionary(data):
    """

    :param data: a list of problems
    :return: a dictionary of all the words in the questions
    """

    word_to_ix = {}
    for question, answer in data:
        for word in question.split():
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    return word_to_ix

# def createTagDictionary(data):
#     tag_to_idx = {}
#     for question, answers in data:
#         for i, answer in enumerate(answers):
#             if answer not in tag_to_idx:
#                 tag_to_idx[answer] = i
#     return tag_to_idx

#tag_to_ix = createTagDictionary(trainingData)

tag_to_ix = {i: i for i in range(100)}
word_to_ix = createWordDictionary(trainingData)

def prepare_question(seq, to_ix):
    """

        :param seq: the list of words in a sentence
        :param to_ix: word_to_ix
        :return: tensor of all the indices of the words
        """

    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


#print(prepare_question(trainingData[0][0].split(), word_to_ix))
#print(createTagDictionary(trainingData))

class LSTMmath(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMmath, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, question):
        embeds = self.word_embeddings(question)
        #print("embeds: {0}".format(embeds))
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(question), 1, -1), self.hidden)
        lstm_out = lstm_out.view(len(question), self.hidden_dim)
        print("lstm_out1: {0}".format(lstm_out))
        forAverage = autograd.Variable(torch.FloatTensor(1, len(question)).fill_(1./len(question)))
        lstm_out = torch.mm(forAverage, lstm_out)
        print("lstm_out2: {0}".format(lstm_out))
        #tag_space = self.hidden2tag(lstm_out.view(len(question), -1))
        tag_space = self.hidden2tag(lstm_out.view(1, -1))
        #print("tag_space: {0}".format(tag_space))
        #we don't need it to be specific to each word. need 1x100 matrix
        tag_scores = F.log_softmax(tag_space)
        return tag_scores

model = LSTMmath(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
inputs = prepare_question(trainingData[0][0].split(), word_to_ix)
tag_scores = model(inputs)
#print(tag_scores)

for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in trainingData:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Variables of word indices.
        sentence_in = prepare_question(sentence.split(), word_to_ix)
        #targets = prepare_question(tags, tag_to_ix)
        #print("tagcrazy: {0}".format(torch.LongTensor(tags)))
        targets = autograd.Variable(torch.LongTensor([tags]))

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

# See what the scores are after training
# for i in range(len(trainingData)):
#     inputs = prepare_question(trainingData[i][0].split(), word_to_ix)
#     tag_scores = model(inputs)
#     #print(tag_scores)
#     value, index = torch.max(tag_scores, 1)
#     print(index)


# The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
#  for word i. The predicted tag is the maximum scoring tag.
# Here, we can see the predicted sequence below is 0 1 2 0 1
# since 0 is index of the maximum value of row 1,
# 1 is the index of maximum value of row 2, etc.
# Which is DET NOUN VERB DET NOUN, the correct sequence!
#print(tag_scores)