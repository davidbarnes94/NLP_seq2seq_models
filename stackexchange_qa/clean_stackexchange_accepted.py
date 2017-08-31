from xml.etree.ElementTree import iterparse
import re
from bs4 import BeautifulSoup as BS
from markdown import markdown
import sys
import os

### TODO: add in user info

encoding = "utf-8"
SAMPLE_SIZE = 1000 # More than 300,000 posts
direc = "data_accepted"

posts_file = "math.stackexchange.com/Posts.xml"
comments_file = "math.stackexchange.com/Comments.xml"

def clean_markdown(raw):
    cleaner = BS(markdown(raw), 'html5lib').get_text() 
    regex = '@\w+|\\n'
    clean_text = re.sub(regex, '', cleaner)
    return clean_text

def extract_posts(posts_file, output_filename=direc+"/posts.txt"):
    """
    Creates an organized text file containing all posts with relevant features.
    If a line contains a question, it has the following format:
        <Post ID#>\t<Post Title>\t<Post Body>\t<Post Score>\n
    If a line contains an answer, it has the following format:
        <Post ID#>\t<Parent Post ID#>\t<Post Body>\t<Post Score>\n
    """
    if not os.path.exists(output_filename.split("/")[0]):
        os.makedirs(output_filename.split("/")[0])

    print("Extracting posts from " + posts_file + "...")
    posts_dict = {}
    with open(output_filename, 'w', encoding=encoding) as f:
        current = 0
        for event, child in iterparse(posts_file, events=('start', 'end')):
            if current > SAMPLE_SIZE:
                break
            elif len(child.attrib) > 0 and event == "start":
                line = ""
                if child.attrib['PostTypeId'] == '1' and 'AcceptedAnswerId' in child.attrib:
                    posts_dict[child.attrib['Id']] = {'accepted': child.attrib['AcceptedAnswerId'], 'other': []}
                    clean_title = clean_markdown(child.attrib['Title'])
                    clean_body = clean_markdown(child.attrib['Body'])
                    line = child.attrib['Id'] + "\t" + clean_title + "\t" + clean_body + "\t" + child.attrib['Score'] + "\n"
                    current += 1
                elif child.attrib['PostTypeId'] == '2':
                    if child.attrib['ParentId'] in posts_dict and not child.attrib['Id'] == posts_dict[child.attrib['ParentId']]['accepted']:
                        posts_dict[child.attrib['ParentId']]['other'].append(child.attrib['Id'])
                    clean_body = clean_markdown(child.attrib['Body'])
                    line = child.attrib['Id'] + "\t" + child.attrib['ParentId'] + "\t" + clean_body + "\t" + child.attrib['Score'] + "\n"
                    current += 1
                f.write(line)
                print_progress(current, SAMPLE_SIZE)
    print("\nFinished extracting posts from " + output_filename + ".\n")
    return posts_dict

def extract_comments(comments_file, output_filename=direc+"/comments.txt"):
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

def create_training_set(posts_dict, output_filename=direc+"/training_without_comments.txt"):
    """
    Creates text file containing the training set of questions and answers
    without comments where each line has the following format:
        <Question ID#>\t<Accepted answer ID#> <Other answer ID#> <Other answer ID#>\n
    """
    print("Creating training set without comments...")
    with open(output_filename, 'w') as f:
        total = len(posts_dict)
        print("# of questions: " + str(total))
        current = 0
        for question in posts_dict:
            line = question + "\t" + posts_dict[question]['accepted'] 
            if len(posts_dict[question]['other']) > 0:
                line += " " + " ".join(posts_dict[question]['other'])
            line += "\n"
            f.write(line)

            current += 1
            print_progress(current, total)
    print("\nFinished creating training set without comments.\n")

def create_training_set_with_comments(posts_dict, comments_dict, output_filename=direc+"/training_with_comments.txt"):
    """
    Creates text file containing the training set of questions and answers
    without comments where each line has the following format:
        <Question ID#> <Question comment ID#> <Question comment ID#>\t
        <Accepted answer ID#> <Accepted answer comment ID#> <Accepted answer comment ID#>\t
        <Other answer ID#'s> <Other answer comment ID#> <Other answer comment ID#>\n
    """
    print("Creating training set with comments...")
    with open(output_filename, 'w') as f:
        total = len(posts_dict)
        print("# of questions: " + str(total))
        current = 0
        for question in posts_dict:
            accepted = posts_dict[question]['accepted']
            others = posts_dict[question]['other']
            line = question
            if question in comments_dict:
                line += " " + " ".join(comments_dict[question])
            
            line += "\t" + accepted
            if accepted in comments_dict:
                line += " " + " ".join(comments_dict[accepted])
            
            for other in others:
                line += "\t" + other
                if other in comments_dict:
                    line += " " + " ".join(comments_dict[other])
            line += "\n"
            f.write(line)

            current += 1
            print_progress(current, total)
    print("\nFinished creating training set with comments.\n")

def print_progress(current, total):
    progress = current/total*100
    sys.stdout.write('\r[{0}] {1}%'.format('#'*int(progress/5), int(progress)))

if __name__ == "__main__":
    qa_dict = extract_posts(posts_file)
    create_training_set(qa_dict)
    
    com_dict = extract_comments(comments_file)
    create_training_set_with_comments(qa_dict, com_dict)
    