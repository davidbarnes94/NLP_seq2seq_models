import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import re
import numpy as np
from random import shuffle
import sys
import pickle
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()

# TODO: prepare for user info
def postsToDict(posts_file):
	""" Reads the extracted posts_file into a usable Python dictionary.
	Parameters:
		posts_file
	Returns:
		posts_dict
	"""
	posts_dict = {}
	with open(posts_file) as f:
		for line in f.readlines():
			vals = line.split("\t")
			if vals[1].isdigit():
				# If post is answer
				posts_dict[int(vals[0])] = {'Parent ID': int(vals[1]),
											'Body': vals[2],
											'Score': int(vals[3])}
			else:
				# If post is question
				posts_dict[int(vals[0])] = {'Title': vals[1],
											'Body': vals[2],
											'Score': int(vals[3])}
	return posts_dict

def splitTrainingData(training_data, ratio=0.2):
	""" Splits the training data into a training set and
		validation set based on the ratio specified.
	Parameters:
		training_data   original data
		ratio           percentage of training data to be test set
	Returns:
		training set, test set
	"""
	shuffle(training_data)
	split_index = int(len(training_data) * (1 - ratio))
	return training_data[:split_index], training_data[split_index:]

def mix_accepted_answer_idx(data):
	'''
	Moves the accepted answer from the front of the list to the back
	:param data:
	:return:
	'''

	new_data = []
	for qtitle, qbody, qscore, answers in data:
		new_answers = answers[1:] + answers[:1]
		new_data.append((qtitle, qbody, qscore, new_answers))
	return new_data

def get_data_with_multiple_answers(data, num_answers = 5):
	""" Reduces dataset to those with more than the number of answers specified

	"""
	new_data = []
	for d in data:
		if len(d[3]) >= num_answers:
			new_data.append(d)
	return new_data

#######################################
### For predicting accepted answers ###
#######################################

def createAcceptedTrainingData(training_file, posts_file="data_accepted/posts.txt", with_comments=False):
	""" Creates a list of training data containing questions title,
		body, score, and answer body and scores.
	Parameters:
		training_file   file containing training data
		with_comments 	True if using training data with comments
	Returns
		training_data   list of tuples
	"""
	posts_dict = postsToDict(posts_file)
	
	training_data = []
	with open(training_file) as f:
		for line in f.readlines():
			vals = line.split()
			question_id = int(vals[0])
			question_title = posts_dict[question_id]['Title']
			question_body = posts_dict[question_id]['Body']
			question_score = posts_dict[question_id]['Score']
			answers = []
			for answer_id in vals[1:]:
				if int(answer_id) not in posts_dict:
					break 
				answer_body = posts_dict[int(answer_id)]['Body']
				answer_score = posts_dict[int(answer_id)]['Score']
				answers.append((answer_body, answer_score))

			# Prevent data points without answers
			if len(answers) == 0:
				continue

			# Randomize index of accepted answer
			
			training_data.append((question_title, question_body, question_score, answers))
	return training_data

# NOTE: when creating vocab, check if word.lower() is in vocab, not just word
def createQuestionVocab(data, raw=True):
	""" Creates a dictionary containing all the vocab in the questions.
	Parameters:
		data 	list of tuples of training data
		raw		True if including all space-separated words
	Returns:
		vocab 	dictionary containing all space-separated lowercase words as keys and index as values
	"""
	vocab = {}
	for qtitle, qbody, qscore, answers in data:
		words = []
		if raw:
			words = qtitle.split() + qbody.split()
		else:
			words = re.findall("[\\\\]*[\w']+", qtitle) + re.findall("[\\\\]*[\w']+", qbody)

		for word in words:
			if word.lower() not in vocab:
				vocab[word.lower()] = len(vocab)
		if qscore not in vocab:
			vocab[qscore] = len(vocab)
	return vocab

def createAnswerVocab(data, raw=True):
	""" Creates a dictionary containing all the vocab in the answers.
	Parameters:
		data 	list of tuples of training data
		raw		True if including all space-separated words
	Returns:
		vocab 	dictionary containing all space-separated lowercase words as keys and index as values
	"""
	vocab = {}
	for qtitle, qbody, qscore, answers in data:
		answers_str = " ".join([answer[0] for answer in answers]) 
		words = []
		if raw:
			words = answers_str.split()
		else:
			words = re.findall("[\\\\]*[\w']+", answers_str)
		# Check if need to convert numbers to floats
		for word in words:
			if word.lower() not in vocab:
				vocab[word.lower()] = len(vocab)
		for answer_score in [answer[1] for answer in answers]:
			if answer_score not in vocab:
				vocab[answer_score] = len(vocab)
	return vocab

#####################################
### Custom-defined PyTorch Models ###
#####################################

class QuestionRNN(nn.Module):
	def __init__(self, input_size, hidden_size, n_layers=1):
		super(QuestionRNN, self).__init__()
		self.n_layers = n_layers # Number of hidden layers in the LSTM
		self.hidden_size = hidden_size # Dimension of a hidden vector

		self.embedding = nn.Embedding(input_size, hidden_size) # To create an embedding for each word
		self.gru = nn.GRU(hidden_size, hidden_size) # The hidden layer

	def forward(self, input, hidden):
		# Input is a LongTensor of the corresponding to a word in the input sequence
		embedded = self.embedding(input).view(1, 1, -1) # Reshape 1 x 1 x hidden_size tensor
		output = embedded
		for i in range(self.n_layers):
			output, hidden = self.gru(output, hidden)
		return output, hidden

	def initHidden(self):
		result = autograd.Variable(torch.zeros(1, 1, self.hidden_size))
		if use_cuda:
			return result.cuda()
		else:
			return result

class AnswerRNN(nn.Module):
	def __init__(self, input_size, hidden_size, n_layers=1):
		super(AnswerRNN, self).__init__()
		self.n_layers = n_layers  # Number of hidden layers in the LSTM
		self.hidden_size = hidden_size  # The dimension of a hidden vector
		self.output2tag = nn.Linear(HIDDEN_DIM, 2)
		self.softmax = nn.LogSoftmax()

		self.embedding = nn.Embedding(input_size, hidden_size)  # to create an embedding for each word
		self.gru = nn.GRU(hidden_size, hidden_size)  # the hidden layer

	def forward(self, input, hidden):
		# Input is a LongTensor of the corresponding to a word in the input sequence
		embedded = self.embedding(input).view(1, 1, -1)  # Reshape 1 x 1 x hidden_size tensor
		output = embedded
		for i in range(self.n_layers):
			output, hidden = self.gru(output, hidden)
		softmax_layer = self.softmax(self.output2tag(output[0]))

		return softmax_layer, hidden

	def initHidden(self):
		result = autograd.Variable(torch.zeros(1, 1, self.hidden_size))
		if use_cuda:
			return result.cuda()
		else:
			return result

def plot_gradient(gradient_norms, num_time_steps, model_name):
    '''
    :param gradient_norms: list of the norm of the gradient for each time step
    :param num_time_steps: total time steps
    :param model_name: name of RNN
    :return:
    '''
    norms = np.array(gradient_norms)
    time_steps = np.arange(num_time_steps)
    plt.title(model_name)
    plt.plot(time_steps, norms)
    plt.show()

def prepare_question_data(question, vocab):
	""" Prepares the question to be fed into the model by converting it into a PyTorch Variable.
	Parameters:
		question 	tuple containing question title, body, and score
		vocab 		dictionary of question vocab
	Returns:
		tensor of all the indices of the question
	"""
	idxs = [vocab[w.lower()] for w in question[0].split()] \
		 + [vocab[w.lower()] for w in question[1].split()] \
		 + [vocab[question[2]]] 
	tensor = torch.LongTensor(idxs)
	return autograd.Variable(tensor)

def prepare_answer_data(answer, vocab):
	""" Prepares the answer to be fed into the model by converting it into a PyTorch Variable.
	Parameters:
		answer 	tuple containing answer body and score
		vocab 	dictionary of answer vocab
	Returns:
		tensor of all the indices of the words
	"""
	idxs = [vocab[w.lower()] for w in answer[0].split()] + [vocab[answer[1]]] 
	tensor = torch.LongTensor(idxs)
	return autograd.Variable(tensor)

def process_question(question, question_model, is_training):
	""" Processes the question by feeding it through the question model.
	Parameters:
		question 			tuple containing question title, body, and score
		question_model		instance of QuestionRNN
	Returns:
		question_outputs	tensor of the final output for each word in the question
		question_hidden 	the last hidden state
	"""

	# Initialize hidden state of first RNN
	question_hidden = question_model.initHidden()

	if is_training:
		question_in = prepare_question_data(question, train_question_vocab)
	else:
		question_in = prepare_question_data(question, test_question_vocab)

	# Length of question outputs?
	question_outputs = autograd.Variable(torch.zeros(len(question_in), question_model.hidden_size))  # Store the final output for each word
	question_outputs = question_outputs.cuda() if use_cuda else question_outputs

	# Enter one word at a time into the model to obtain hidden and output states
	for i in range(len(question_in)):
		question_output, question_hidden = question_model(question_in[i], question_hidden)
		question_outputs[i] = question_output[0][0]

	return question_outputs, question_hidden

def process_answer(answer, answer_model, question_final_hidden, is_training):
	""" Processes the answer by feeding it through the answer model.
	Parameters:
		answer 					tuple containing answer body and score
		answer_model			instance of AnswerRNN
		question_final_hidden	last hidden state from the question RNN
	Returns:
		answer_outputs 			tensor of the final output for each word in the answer
	"""
	if is_training:
		answer_in = prepare_answer_data(answer, train_answer_vocab)
	else:
		answer_in = prepare_answer_data(answer, test_answer_vocab)

	answer_hidden = question_final_hidden # Last hidden state from the question becomes the initial hidden state of the answer model

	# Enter one word at a time into the model to obtain hidden and output states
	for i in range(len(answer_in)):
		#answer_output, answer_hidden = answer_model(answer_in[i], answer_hidden)
		#answer_outputs[i] = answer_output[0][0]
		softmax_output, answer_hidden = answer_model(answer_in[i], answer_hidden)

	return softmax_output

def create_models():
	""" Creates a QuestionRNN and question optimizer to process the question 
		and an AnswerRNN and answer optimizer to process answers
	Returns:
		question_model 		an instance of QuestionRNN
		question_optimizer 	an optimizer for the parameters of the question_model
		answer_model 		an instance of AnswerRNN
		answer_optimizer	an optimizer for the parameters of the answer_model
	"""
	question_model = QuestionRNN(len(train_question_vocab), HIDDEN_DIM)
	answer_model = AnswerRNN(len(train_answer_vocab), HIDDEN_DIM)

	question_optimizer = optim.SGD(question_model.parameters(), lr=0.1)
	answer_optimizer = optim.SGD(answer_model.parameters(), lr=0.1)

	return question_model, question_optimizer, answer_model, answer_optimizer

def train(training_data, loss_function, epochs = 100):
	""" Trains the models on the training data using the loss function specified 
		for a number of epochs specified
	Parameters:
		training_data	list of training data containing tuples of questions and corresponding answers
		loss_function	loss function to be used when training
		epochs			number of epochs
	Returns:
		question_model	trained QuestionRNN
		answer_models	trained AnswerRNN
	"""

	question_model, question_optimizer, answer_model, answer_optimizer = create_models()

	print("Training model for %d epochs." % epochs)

	e = 0
	gradient_norms = []
	gradient_norms_question = []
	params = list(answer_model.parameters())
	params_question = list(question_model.parameters())
	for epoch in range(epochs):
		for data in training_data:

			print_progress(e+1, len(training_data)*epochs)
			e += 1

			question = (data[0], data[1], data[2])
			answers = data[3]


			# Fix when there are no answers
			if len(answers) == 0:
				continue

			# Set all gradients to zero
			question_model.zero_grad()
			answer_model.zero_grad()

			# To store the final ouput from the answer RNN
			predicted_tags = autograd.Variable(torch.zeros(len(answers), 2))

			# Feed the question through the question RNN
			question_outputs, last_hidden = process_question(question, question_model, True)


			# Feed each answer through the answer RNN
			for i, answer in enumerate(answers):
				# TODO: ???
				#answer_outputs[i] = process_answer(answer, answer_model, last_hidden, True)[-1].view(1, -1)
				predicted_tags[i] = process_answer(answer, answer_model, last_hidden, True)


			# Each answer RNN outputs a softmax over 0 and 1
			# 0 - incorrect answer
			# 1 - correct answer
			#predicted_tags, true_tags = predict_answer(len(answers)-1, answer_outputs)
			true_tags = autograd.Variable(torch.zeros(len(answers)))
			true_tags[0] = 1


			loss = loss_function(predicted_tags, true_tags.long())
			loss.backward()


			question_optimizer.step()
			answer_optimizer.step()
			gradient_norms.append(params[0].grad.data.norm(2))
			gradient_norms_question.append(params_question[0].grad.data.norm(2))


	plot_gradient(gradient_norms, len(training_data)*epochs, "answer model")
	plot_gradient(gradient_norms_question, len(training_data)*epochs, 'question_model')

	print("\nFinished training model.")
	return question_model, answer_model

# TODO: Check if having index=0 as accepted answer affects training
# with index=0, the performance is basically the same


def predict_accepted_answer_index(predicted_tags):
	"""
	Parameters:
		predicted_tags 	tensor with each row being the log softmax over 0 and 1 for belief in the correctness of that answer
		ans_index 		index of the correct answer
	Returns:
		index of the predicted accepted answer

	Method for predicting the correct answer:
	- an answer will be considered for possibly being the correct response if its log softmax value for 1 is larger than that of 0
	- from the list of possible answers, the one with the highest log softmax value for 1 will be chosen
	- otherwise, no answer is chosen
	"""
	predicted_tags = predicted_tags.data.numpy()
	max_ones = [-100] * len(predicted_tags)
	for i, tag_scores in enumerate(predicted_tags):
		if tag_scores[1] > tag_scores[0]:
			max_ones[i] = tag_scores[1]

	max_one = max_ones.index(max(max_ones))


	return max_one

def test(question_model, answer_model, test_data, is_training=False):
	"""
	Parameters:
		question_model	trained RNN for processing the question
		answer_models	trained answer RNN's for processing the answers
		data 			data for testing the model
		is_training 	True if the training data is used
	Returns:
		accuracy
	"""
	if is_training:
		print("\nTesting model on training set.")
	else:
		print("\nTesting model on test set.")

	num_correct = 0

	for qi, data in enumerate(test_data):
		question = (data[0], data[1], data[2])
		answers = data[3]
		predicted_tags = autograd.Variable(torch.zeros(len(answers), 2))

		# Fix when there are no answers
		if len(answers) == 0:
			print("Question: {0} has no answers in data so skipping...".format(qi))
			continue

		question_in, last_hidden = process_question(question, question_model, is_training)
		for i, answer in enumerate(answers):
			# TODO: ???
			predicted_tags[i] = process_answer(answer, answer_model, last_hidden, is_training)

		#predicted_tags, true_tags = predict_answer(len(answers)-1, answer_outputs)

		predicted_index = predict_accepted_answer_index(predicted_tags)
		#num_correct += int(predicted_index == len(answers)-1)
		num_correct += int(predicted_index == 0)

		if not predicted_index == -1:
			print("Question: {0}, Correct answer index: {1}, Predicted_answer: {2}".format(qi, 0, predicted_index))
		else:
			print("Question: {0}, Correct answer index: {1}, Model doesn't think any of the answers are accepted".format(qi, 0))

	return "The model correctly predicted {0} out of {1} questions".format(num_correct, len(test_data))



def print_progress(current, total):
	""" Prints an in-line progress bar in the terminal
	Parameters:
		current		current number of iterations
		total		total number of iterations
	"""
	erase = '\x1b[2K'
	progress = current/total*100
	sys.stdout.write(erase + '[{0}] {1}%    {2}/{3}\r'.format('#'*int(progress/5), int(progress), current, total))


#####################################
### Saving/loading pickle objects ###
#####################################

def save_models(question_model, answer_model, q_pickle_path="models/question_model.pkl", a_pickle_path="models/answer_model.pkl"):
	""" Saves the question and answer model to pickle files
	Parameters:
		question_model 	QuestionRNN instance
		answer_model 	AnswerRNN instance
		q_pickle_path	pickle file path for the question_model
		a_pickle_path	pickle file path for the answer_model
	"""
	print("\nSaving models from pickle file...")
	with open(q_pickle_path, 'wb') as qf:
		pickle.dump(question_model, qf)

	with open(a_pickle_path, 'wb') as af:
		pickle.dump(answer_model, af)

def load_models(q_pickle_path="models/question_model.pkl", a_pickle_path="models/answer_model.pkl"):
	""" Loads the question and answer models from pickle files
	Parameters:
		q_pickle_path	pickle file path for the question_model
		a_pickle_path	pickle file path for the answer_model
	Returns:
		question_model 	QuestionRNN instance
		answer_model 	AnswerRNN instance
	"""
	print("\nLoading models from pickle file...")
	with open(q_pickle_path, 'rb') as qf:
		question_model = pickle.load(qf)

	with open(a_pickle_path, 'rb') as af:
		answer_model = pickle.load(af)
	return question_model, answer_model

def save_pickle_object(pickle_obj, pickle_file):
	""" Saves objects to pickle files
	Parameters:
		pickle_obj 		python object to be saved
		pickle_file 	pickle file path for the pickle_obj
	"""
	print("\nSaving object to " + pickle_file)
	with open("pickles/" + pickle_file, 'wb') as pf:
		pickle.dump(pickle_obj, pf)

def load_pickle_object(pickle_file):
	""" Loads objects from pickle files
	Parameters:
		pickle_file 	pickle file path for the pickle_obj
	Returns:
		pickle_obj 		python object
	"""
	print("\nLoading object from " + pickle_file )
	with open("pickles/" + pickle_file, 'rb') as pf:
		obj = pickle.load(pf)
	return obj

#################################################################

if __name__ == '__main__':
	HIDDEN_DIM = 256

	posts_file ="data_accepted/posts.txt"
	training_file = "data_accepted/training_without_comments.txt"

	loss_function = nn.NLLLoss()

	loading_data = False # Currently the dataset in the pickles folder is a set of 10 questions each with 5+ answers
	loading_model = False

	
	# The accepted answer index is currently the last index

	if not loading_data:
		posts_dict = postsToDict(posts_file)
		training_data = createAcceptedTrainingData(training_file)
		training_data, test_data = splitTrainingData(training_data)

		#training_data = mix_accepted_answer_idx(training_data)
		training_data = get_data_with_multiple_answers(training_data)
		
		save_pickle_object(training_data[:10], "temp_training_data.pkl")
		save_pickle_object(test_data[:10], "temp_test_data.pkl")
	else:
		training_data = load_pickle_object("temp_training_data.pkl")
		test_data = load_pickle_object("temp_test_data.pkl")

	print("Training data size: %d" % len(training_data))
	print("Test data size: %d" % len(test_data))

	train_question_vocab = createQuestionVocab(training_data)
	test_question_vocab = createQuestionVocab(test_data)

	train_answer_vocab = createAnswerVocab(training_data)
	test_answer_vocab = createAnswerVocab(test_data)

	if not loading_model:
		question_model, answer_model = train(training_data, loss_function, 2)
		save_models(question_model, answer_model)
	else:
		question_model, answer_model = load_models()

	accuracy1 = test(question_model, answer_model, training_data, is_training=True)
	print(accuracy1)
	# accuracy2 = test(question_model, answer_model, test_data)
	# print(accuracy2)

