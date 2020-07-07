import logging
import time
import json
import mimetypes
import http.client
import warnings
import re
import spacy

import pandas as pd
import numpy as np

from langdetect import detect, DetectorFactory
from bs4 import BeautifulSoup
from tqdm.autonotebook import tqdm

# Add tqdm to pandas
tqdm.pandas(desc="Preprocess Data")
# Ignore warnings
warnings.filterwarnings('ignore')
# Configure logging
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s",
                    level=logging.INFO)
logger = logging.getLogger()
# Set seed for DetectorFactory
DetectorFactory.seed = 1
# NLP tokenizer and preprocessing
NLP = spacy.load('en_core_web_sm')

def get_regex_expression():
    """
    Generate some regex expression
    """
    # Match non alphanumeric characters
    NON_ALPHANUMERIC_REGEX = r'[^a-zA-Z0-9À-ÿ\u00f1\u00d1\s]'
    # Match any link or url from text
    LINKS_REGEX = r'https?:\/\/.*[\r\n]'
    # Match hashtags
    HASHTAGS_REGEX = r'\#[^\s]*'
    # Match twitter accounts
    TWITTER_ACCOUNTS_REGEX = r'\@[^\s]*'
    # Match Author:
    AUTHOR_REGEX = r'author'
    # Match email
    EMAIL_REGEX = r"\S*@\S+"
    # Group of regex
    MATCHES_GROUPED = ('({})'.format(reg) for reg in [
        LINKS_REGEX, HASHTAGS_REGEX, TWITTER_ACCOUNTS_REGEX, AUTHOR_REGEX,
        EMAIL_REGEX, NON_ALPHANUMERIC_REGEX
    ])

    # Regex for matches group
    MATCHES_GROUPED_REGEX = r'{}'.format(('|'.join(MATCHES_GROUPED)))

    return MATCHES_GROUPED_REGEX

REGEX = get_regex_expression()

def remove_unnecesary_text(text, regex):
    """
    Remove unnecesary text using regex

    Args:
        text -- python string
        regex -- python regex
    Returns:
        text -- python string
    """
    return re.sub(regex, ' ', text, flags=re.M | re.I)


# Remove all whitespace characters
def remove_whitespace(text):
    """
    Remove unnecesary whitespace

    Args:
        text -- python string
    Returns:
        text -- python string
    """
    return ' '.join(text.split())


def preprocess_data(text, removing_stops=False, lemmatize=False):
    """
    Preprocess string data.
    Args:
        text -- A string python that is on the columns of a pandas dataframe
        regex -- Regular expression
        removing_stops -- Boolean python, if True remove english stops words
        lemmatize -- Boolean python, if True lemmatize english words
    Returns:
        text -- The Preprocess string data python
    """
    # Clean text
    text = remove_whitespace(remove_unnecesary_text(text, REGEX))

    # Tokenize the text of the blogs
    tokens = NLP(text)

    # Remove all punctuation marks
    tokens = [token for token in tokens if not token.is_punct]

    # Remove numbers or amount representation
    tokens = [token for token in tokens if not token.like_num]

    if removing_stops:
        # Remove stopswords
        tokens = [token for token in tokens if not token.is_stop]

    if lemmatize:
        # Lemmatize words
        tokens = [token.lemma_.strip().lower() for token in tokens]
    else:
        # Convert to str and lowerize
        tokens = [token.text.strip().lower() for token in tokens]

    tokens = [token for token in tokens if len(token) > 1]

    return " ".join(tokens)


def download_dataset(page):
    """
    Download dataset and save to a python list
    Args:
        page -- last page scarped
    Returns:
        data_temp -- python list containing dict for each blog data
    """
    sw = True
    data_temp = []
    numblog = 0
    while sw:
        try:
            conn = http.client.HTTPSConnection("koombeastaging.wpengine.com")
            conn.request("GET",
                         f"//wp-json/wp/v2/posts?page={page}&per_page=1")
            res = conn.getresponse()
            data = res.read()
            data = json.loads(data)
            numblog += len(data)
            data_temp = data_temp + data
            page += 1
            if numblog % 20 == 0:
                logger.info("Downloading blogs = {0}".format(numblog))
                time.sleep(2)
        except Exception as e:
            logger.error("Error! {0}".format(e))
            sw = False
    last_page = page - 1
    return data_temp, last_page


def clean_html(html_content):
    """
    Clean html form of the data
    Argument:
        html_content -- Blog's content in html form

    Returns:
        clean_text -- python string containing the blog's
        content cleaned and parsed with the beatifulsoup html.parser method
    """

    clean_text = None
    soup = BeautifulSoup(html_content, "html.parser")
    clean_text = soup.get_text()
    return clean_text


def get_data_frame(page):
    """
    Clean the data and generate a pandas dataframe with the values
    Args:
        page -- last page scrapped
    Return:
        df -- pandas dataframe with all the data and sort by id
    """
    logger.info("Downloading Dataset on {0}/{1}".format(
        "koombeastaging.wpengine.com", "//wp-json/wp/v2/posts?page&per_page"))
    data_temp, last_page = download_dataset(page)
    logger.info(
        "Begin To clean datablogs and grab title and content information")

    # Clean html form of data blogs
    blogs = []
    for blog in tqdm(data_temp, desc="Cleaning html data"):
        info_blog = {}
        info_blog["id"] = blog["id"]
        info_blog["title"] = clean_html(blog["title"]["rendered"])
        info_blog["content"] = clean_html(blog["content"]["rendered"])
        info_blog["slug"] = clean_html(blog["slug"])
        blogs.append(info_blog)

    # Transform to a simple dataframe
    df = pd.DataFrame(blogs)
    idx_ord = df.id.sort_values(ascending=True).index
    df = df.loc[idx_ord]
    df.reset_index(drop=True, inplace=True)
    logger.info("Finish!! Donwloading the blogs")

    return df, last_page


def data_language_detection(data):
    """detect the language of the data

    Args:
        data (String): blog's data string

    Returns:
        lang (String): language code of the data
    """
    return detect(data)