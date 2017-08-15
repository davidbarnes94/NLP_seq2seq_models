# def predict_answer(ans_index, answer_outputs):
#     true_tags = autograd.Variable(torch.zeros(NUM_ANSWERS))
#     tag_space = autograd.Variable(torch.zeros(NUM_ANSWERS, 2))
#     for i in range(len(answer_outputs)):
#         true_tags[i] = 1 if ans_index == i else 0
#         tag_space[i] = output2tag(answer_outputs[i].view(1, -1))
#
#     m = nn.LogSoftmax()
#     predicted_tag_scores = m(tag_space)
#     print(predicted_tag_scores)
#     print(true_tags)
#     return predicted_tag_scores, true_tags

# def predict_answer(ans_index, answer_outputs):
#     true_tags = [0]*NUM_ANSWERS
#     tag_space = []
#     print(true_tags)
#     print(tag_space)
#     for i in range(len(answer_outputs)):
#         print("i: {0}".format(i))
#         true_tags[i] = 1 if ans_index == i else 0
#         tag_space.append(output2tag(answer_outputs[i].view(1, -1)))
#         print("true_tags: {0}".format(true_tags))
#         print("tag_space: {0}".format(tag_space))
#
#     tag_space = autograd.Variable(torch.FloatTensor(tag_space))
#     true_tags = autograd.Variable(torch.LongTensor(true_tags))
#     m = nn.LogSoftmax()
#     predicted_tag_scores = m(tag_space)
#     print(predicted_tag_scores)
#     print(true_tags)
#     return predicted_tag_scores, true_tags

# class AnswerRNN(nn.Module):
#     def __init__(self, hidden_size, output_size, question_embedding, tagset_size=2, n_layers=1, n_answers=4):
#         super(AnswerRNN, self).__init__()
#         self.n_layers = n_layers
#         self.hidden_size = hidden_size
#         self.n_answers = n_answers
#         self.question_embedding = question_embedding
#
#         self.embedding = nn.Embedding(output_size, hidden_size)
#         self.lstm = nn.LSTM(hidden_size, hidden_size)
#         #self.out = nn.Linear(hidden_size, output_size)
#         self.softmax = nn.Softmax()
#         self.hidden2tag = nn.Linear(hidden_size, tagset_size)
#
#
#     # def forward(self, input, hidden):
#     #     last_outputs = []
#     #     for i in range(self.n_answers):
#     #         output = self.embedding(input).view(1, 1, -1)
#     #         for j in range(self.n_layers):
#     #             output = F.relu(output)
#     #             output, hidden = self.lstm(output, hidden)
#     #         last_outputs.append(output[0])
#     #     qa_similarity = [torch.dot(self.question_embedding, output) for output in last_outputs]
#     #     return self.softmax(qa_similarity)
#
#     def forward(self, input, hidden):
#         # the input is an element LongTensor of the corresponding to a word in the input sequence
#         # the hidden is the hidden state from the previous time step
#         output = self.embedding(input).view(1, 1, -1)
#         for i in range(self.n_layers):
#             output = F.relu(output)
#             output, hidden = self.lstm(output, hidden)
#         output = self.softmax(self.out(output[0]))
#         tag_space = self.softmax(self.hidden2tag(output.view(1, -1)))
#         return output, hidden, tag_space
#
#
#     def initHidden(self):
#         result = autograd.Variable(torch.zeros(1, 1, self.hidden_size))
#         if use_cuda:
#             return result.cuda()
#         else:
#             return result

# for epoch in range(10):
#     for question, answers, ans_index in trainingData:
#         #initialize hidden state of first RNN
#         question_hidden = questionModel.initHidden()
#
#         #set all gradients to zero
#         questionModel.zero_grad()
#         answer0Model.zero_grad()
#         answer1Model.zero_grad()
#         answer2Model.zero_grad()
#         answer3Model.zero_grad()
#
#         encoder_outputs = autograd.Variable(torch.zeros(len(word_to_ix), questionModel.hidden_size)) #store the final output for each word
#         encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
#
#         answerInputs = autograd.Variable(torch.zeros(4, answer0Model.hidden_size)) #what does this do?
#
#         #should we call zero_grad on the optimizer or the model?
#         #doesn't matter: https://discuss.pytorch.org/t/zero-grad-optimizer-or-net/1887/4
#
#         #process the question
#         question_in = prepare_data(question.split(), word_to_ix)
#         #print(question, answers[ans_index])
#         #target = autograd.Variable(torch.LongTensor([answers[ans_index]]))
#         #print("question_hidden: {0}".format(question_hidden))
#
#         for word_index in range(len(question_in)):
#             question_output, question_hidden = questionModel(question_in[word_index], question_hidden)
#             encoder_outputs[word_index] = question_output[0][0]
#
#         #print(answers[0])
#         #print(tag_to_ix)
#
#         #process the answers
#         try:
#             answer0_in = prepare_data(answers[0].split(), tag_to_ix)
#         except:
#             answer0_in = prepare_data(answers[0], tag_to_ix)
#
#         #print("answer0_in: {0}".format(answer0_in))
#         answerOutputs = autograd.Variable(torch.zeros(len(answer0_in), answer0Model.hidden_size))
#         answer_hidden = question_hidden #last hidden state from the encoder becomes the initial hidden state of the decoder
#         #print("bat")
#         for word_index in range(len(answer0_in)):
#             #print("cat")
#             answer_output, answer_hidden = answer0Model(answer0_in[word_index], answer_hidden)
#             #print("dog")
#             #print("answer_output: {0}".format(answer_output))
#             answerOutputs[word_index] = answer_output[0][0]
#
#         #print("answerOutputs: {0}".format(answerOutputs))
#
#         #parameter update
#         true_tag = 1 if ans_index == 0 else 0
#         true_tag = autograd.Variable(torch.LongTensor([true_tag]))
#         tag_space = output2tag(answerOutputs[0].view(1, -1))
#         predicted_tag_scores = F.log_softmax(tag_space)
#         #print(predicted_tag_scores)
#
#         #loss = loss_function(modelChoice, autograd.Variable(torch.zeros(4).index_fill_(0, torch.LongTensor([ans_index]), 1)))
#         loss = loss_function(predicted_tag_scores, true_tag)
#         loss.backward()
#
#         questionOptimizer.step()
#         answer0Optimizer.step()
#         answer1Optimizer.step()
#         answer2Optimizer.step()
#         answer3Optimizer.step()
#     #print(epoch)

# #See what the scores are after training
# sumAccuracyTraining = 0
# sumAccuracyTest = 0
# encoder_outputs2 = autograd.Variable(torch.zeros(len(word_to_ix), questionModel.hidden_size)) #store the final output for each word
# encoder_outputs2 = encoder_outputs.cuda() if use_cuda else encoder_outputs
#
# for i in range(len(trainingData)):
#     print("i: {}".format(i))
#     inputs = prepare_data(trainingData[i][0].split(), word_to_ix)
#     for word_index in range(len(inputs)):
#         out, hide = questionModel(inputs[word_index], questionModel.initHidden())
#         encoder_outputs2[word_index] = out[0][0]
#     try:
#         answer0_in2 = prepare_data(trainingData[i][1][0].split(), tag_to_ix)
#     except:
#         answer0_in2 = prepare_data(trainingData[i][1][0], tag_to_ix)
#
#     answerOutputs2 = autograd.Variable(torch.zeros(len(answer0_in2), answer0Model.hidden_size))
#     answer_hidden2 = out #last hidden state from the encoder becomes the initial hidden state of the decoder
#     #print("bat")
#     for word_index in range(len(answer0_in2)):
#         print("word_index: {}".format(word_index))
#         #print("cat")
#         answer_output2, answer_hidden2 = answer0Model(answer0_in2[word_index], answer_hidden2)
#         #print("dog")
#         print("answer_output: {0}".format(answer_output2))
#         answerOutputs2[word_index] = answer_output2[0][0]
#
#     tag_scores = F.log_softmax(output2tag(answerOutputs2[0].view(1, -1)))
#     value, index = torch.max(tag_scores, 1)
#     sumAccuracyTraining += torch.eq(index.data, torch.LongTensor([trainingData[i][2]]))
#

# print("total correct: {0}".format(sumAccuracyTraining))
# accuracyTraining = sumAccuracyTraining.numpy()/float(len(trainingData))
# print("percentage training accuracy: {0}".format(accuracyTraining))

# testDataDict = createWordDictionary(testData)
# for i in range(len(testData)):
#     inputs = prepare_question(testData[i][0].split(), testDataDict)
#     tag_scores = model(inputs)
#     value, index = torch.max(tag_scores, 1)
#     sumAccuracyTest += torch.eq(index.data, torch.LongTensor([testData[i][1]]))
#
# print("total correct: {0}".format(sumAccuracyTest))
# accuracyTest = sumAccuracyTest.numpy()/float(len(testData))
# print("percentage test accuracy: {0}".format(accuracyTest))
