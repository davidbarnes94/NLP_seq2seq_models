# StackExchange Q&A Predictive Seq2Seq Model

Building a Seq2Seq model.

## File Contents and Purpose:

* **clean_stackexchange_accepted.py** - extracts the stackexchange posts.xml and comments.xml files into data_accepted/ directory
* **clean_stackexchange_rankings.py** - extracts the stackexchange posts.xml and comments.xml files into data_rankings/ directory
* **seq2seq_accepted_model.py** - runs the seq2seq model on the specified set of data (currently working on model for accepted answers without comments)
* **data_accepted/** - contains:
	* posts.txt - sample of 1,000 posts (roughly 200 questions) in the following format:
	```
	For question posts: <Post ID#>\t<Post Title>\t<Post Body>\t<Post Score>\n
	For answer posts: <Post ID#>\t<Parent Post ID#>\t<Post Body>\t<Post Score>\n
	```
	* comments.txt -  contains comments in the format:
	```
	<Comment ID#>\t<Parent Post ID#>\t<Comment Text>\t<Comment Score>\n
	```
	* training_with_comments.txt - contains training set with tags for correspoding comments in the following format (multiple lines to make reading easier):
	```
	<Question ID#> <Question comment ID#> <Question comment ID#>\t
	<Accepted answer ID#> <Accepted answer comment ID#> <Accepted answer comment ID#>\t
	<Other answer ID#'s> <Other answer comment ID#> <Other answer comment ID#>\n
	```
	* training_without_comments.txt - contains training set in the following format:
	```
	<Question ID#>\t<Accepted answer ID#> <Other answer ID#> <Other answer ID#>\n
	```
* **models/** - directory to save/load models
* **pickles/** - directory to save/load other objects
	* temp_test_data.pkl - saved list of 10 test data points with 5+ answers each
	* temp_training_data.pkl - saved list of 10 training data points with 5+ answers each

## Built With

* [PyTorch](http://pytorch.org/) - Neural Network Framework
