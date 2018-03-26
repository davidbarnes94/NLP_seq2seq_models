from xml.etree.ElementTree import iterparse
import re
from bs4 import BeautifulSoup as BS
from markdown import markdown
import sys
import os

encoding = "utf-8"
SAMPLE_SIZE = 5000 # More than 300,000 posts

posts_file = "math.stackexchange.com/Posts.xml"
comments_file = "math.stackexchange.com/Comments.xml"

def clean_markdown(raw):
    cleaner = BS(markdown(raw), 'html5lib').get_text() 
    regex = '@\w+|\\n'
    clean_text = re.sub(regex, '', cleaner)
    return clean_text

def extract_posts(posts_file, output_filename="data_rankings/posts.txt"):
    """
    Creates an organized text file containing all posts with relevant features.
    If a line contains a question, it has the following format:
        <Post ID#>\t<Post Title>\t<Post Body>\t<Post Score>\n
    If a line contains an answer, it has the following format:
        <Post ID#>\t<Parent Post ID#>\t<Post Body>\t<Post Score>\n

    Returns:
        posts_dict  dictionary where keys are question ID#'s and values are sorted lists
                    of tuples containing the answer ID#'s and score
    """
    if not os.path.exists(output_filename.split("/")[0]):
        os.makedirs(output_filename.split("/")[0])

    print("Extracting posts from " + output_filename + "...")
    posts_dict = {}
    with open(output_filename, 'w', encoding=encoding) as f:
        current = 0
        for event, child in iterparse(posts_file, events=('start', 'end')):
            if current > SAMPLE_SIZE:
                break
            elif len(child.attrib) > 0 and event == "start":
                line = ""
                if child.attrib['PostTypeId'] == '1':
                    if child.attrib['Id'] not in posts_dict:
                        posts_dict[child.attrib['Id']] = []
                    clean_title = clean_markdown(child.attrib['Title'])
                    clean_body = clean_markdown(child.attrib['Body'])
                    line = child.attrib['Id'] + "\t" + clean_title + "\t" + clean_body + "\t" + child.attrib['Score'] + "\n"
                elif child.attrib['PostTypeId'] == '2':
                    if child.attrib['ParentId'] not in posts_dict:
                        posts_dict[child.attrib['ParentId']] = []
                    insert_into_sorted(posts_dict[child.attrib['ParentId']], (child.attrib['Id'], int(child.attrib['Score'])))
                    clean_body = clean_markdown(child.attrib['Body'])
                    line = child.attrib['Id'] + "\t" + child.attrib['ParentId'] + "\t" + clean_body + "\t" + child.attrib['Score'] + "\n"
                f.write(line)

                current += 1
                print_progress(current, SAMPLE_SIZE)
    print("\nFinished extracting posts from " + output_filename + ".\n")
    return posts_dict

def extract_comments(comments_file, output_filename="data_rankings/comments.txt"):
    """
    Creates an organized text file containing all comments with relevant features.
    Each line has the following format:
        <Comment ID#>\t<Parent Post ID#>\t<Comment Text>\t<Comment Score>\n
    """
    if not os.path.exists(output_filename.split("/")[0]):
        os.makedirs(output_filename.split("/")[0])

    print("Extracting comments from " + comments_file + "...")
    comments_dict = {}
    with open(output_filename, "w", encoding=encoding) as f:
        current = 0
        for event, child in iterparse(comments_file, events=('start', 'end')):
            if current > SAMPLE_SIZE:
                break
            elif len(child.attrib) > 0 and event == "start":
                if child.attrib['PostId'] not in comments_dict:
                    comments_dict[child.attrib['PostId']] = []
                comments_dict[child.attrib['PostId']].append(child.attrib['Id'])
                clean_comment = clean_markdown(child.attrib['Text'])
                line = child.attrib['Id'] + "\t" + child.attrib['PostId'] + "\t" + clean_comment + "\t" + child.attrib['Score'] + "\n"
                f.write(line)

                current += 1
                print_progress(current, SAMPLE_SIZE)
    print("\nFinished extracting comments from " + comments_file + ".\n")
    return comments_dict

def create_training_set(posts_dict, output_filename="data_rankings/training_without_comments.txt"):
    """
    Creates text file containing the training set of questions and answers
    without comments where each line has the question ID# followed by a sorted
    list of answer ID#'s from highest score to lowest score:
        <Question ID#>\t<Answer ID#>\<Answer Score> <Answer ID#>\<Answer Score>\n
    """
    print("Creating training set without comments...")
    with open(output_filename, 'w') as f:
        total = len(posts_dict)
        print("# of questions: " + str(total))
        current = 0
        for question in posts_dict:
            answers = list(map(lambda x: x[0] + "/" + str(x[1]), posts_dict[question]))
            line = question + "\t" + " ".join(answers) + "\n"
            f.write(line)

            current += 1
            print_progress(current, total)
    print("\nFinished creating training set without comments.\n")

def create_training_set_with_comments(posts_dict, comments_dict, output_filename="data_rankings/training_with_comments.txt"):
    """
    Creates text file containing the training set of questions and answers
    without comments where each line has the following format:
        <Question ID#> <Question comment ID#> <Question comment ID#>\t
        <Answer ID#>/<Answer Score> <Answer comment ID#> <Answer comment ID#>\t
        <Answer ID#>/<Answer Score> <Answer comment ID#> <Answer comment ID#>\n
    """
    print("Creating training set with comments...")
    with open(output_filename, 'w') as f:
        total = len(posts_dict)
        print("# of questions: " + str(total))
        current = 0
        for question in posts_dict:
            answers = posts_dict[question]
            line = question
            if question in comments_dict:
                line += " " + " ".join(comments_dict[question])

            for answer, score in answers:
                line += "\t" + answer + "/" + str(score)
                if answer in comments_dict:
                    line += " " + " ".join(comments_dict[answer])
            line += "\n"
            f.write(line)

            current += 1
            print_progress(current, total)
    print("\nFinished creating training set with comments.\n")

def print_progress(current, total):
    progress = current/total*100
    sys.stdout.write('\r[{0}] {1}%'.format('#'*int(progress/5), int(progress)))

def insert_into_sorted(lst, elem):
    i = 0
    while i < len(lst) and elem[1] < lst[i][1]: 
        i += 1
    lst.insert(i, elem)

if __name__ == "__main__":
    qa_dict = extract_posts(posts_file)
    create_training_set(qa_dict)
    
    com_dict = extract_comments(comments_file)
    create_training_set_with_comments(qa_dict, com_dict)
