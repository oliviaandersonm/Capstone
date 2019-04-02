#similar.py
#compares wmd similiarities among sentences or articles
<<<<<<< HEAD
import sys, os, pyemd

#get vectors
from gensim.models import KeyedVectors
wmd_vectors = KeyedVectors.load(os.path.expanduser('~/Capstone/data/wmd/wmd_vectors.kv'), mmap='r')
print('Vectors loaded')
=======
import sys, os, re
import pyemd
import argparse
import similar_article_load


print('Loading Vectors')
from gensim.models import KeyedVectors
wmd_vectors = KeyedVectors.load(os.path.expanduser('~/Capstone/data/wmd/wmd_vectors.kv'), mmap='r')
>>>>>>> 036606f5efa6a891bc0940ece104183b0c021e18

#get stopwords
from nltk.corpus import stopwords
from nltk import download
download('stopwords')
stop_words = stopwords.words('english')

<<<<<<< HEAD
def get_wmd_distance(one, two):
=======

def get_wmd(one, two):
>>>>>>> 036606f5efa6a891bc0940ece104183b0c021e18
    #remove stopwords and lower
    one = one.lower().split()
    two = two.lower().split()
    one = [x for x in one if x not in stop_words]
    two = [x for x in two if x not in stop_words]

    distance = wmd_vectors.wmdistance(one, two)
<<<<<<< HEAD
    return distance

def parse_txt(dic, file):
    url = file.readline().rstrip('\n')
=======

    if distance == float('Inf'):
        distance = 0.0000

    return distance

def parse_txt(dic1, dic2, file):
    url = file.readline().rstrip('\n')
    title = file.readline().rstrip('\n')
    dic2[url] = title
>>>>>>> 036606f5efa6a891bc0940ece104183b0c021e18
    text = ''
    line = file.readline()
    while(line):
        text += line
        line = file.readline()
        if(line == '*****\n'):
<<<<<<< HEAD
            dic[url] = text
            parse_txt(dic, file)

#pull text from requested url (newspaper)
from newspaper import Article
requested = Article(sys.argv[1])
requested.download()
requested.parse()

#parse temp file for sentences/articles
article_dic = {}
with open('temp_sim.txt', 'r') as f:
    parse_txt(article_dic, f)

#clean article text and get wmd distances
distances_dic = {}
for key in article_dic:
    article_dic[key] = article_dic[key].replace('\n', '')
    distances_dic[key] = get_wmd_distance(requested.text, article_dic[key])


#get most similar articles
not_top_five = sorted(distances_dic.values())
not_top_five = not_top_five[5:]
top_five = {key:val for key, val in distances_dic.items() if val not in not_top_five}


#function for wmd distance between sentence[i], sentences in file articles
#compare wmd distances and return lowest values
#highlight diffs (bold?)
=======
            dic1[url] = text
            parse_txt(dic1, dic2, file)

def find_top_articles(url):
    #pull text from requested url (newspaper)
    from newspaper import Article
    requested = Article(url)
    requested.download()
    requested.parse()

    #get articles
    print("Finding top five similar articles to \'%s\'" % requested.title)
    similar_article_load.get_similar_articles(url)

    #parse temp file for sentences/articles
    article_dic = {}
    titles_dic = {}
    with open('temp_sim.txt', 'r') as f:
        parse_txt(article_dic, titles_dic, f)

    #clean article text and get wmd distances
    distances_dic = {}
    for key in article_dic:
        article_dic[key] = article_dic[key].replace('\n', '')
        distances_dic[key] = get_wmd(requested.text, article_dic[key])


    #get most similar articles
    not_top_five = sorted(distances_dic.values())
    not_top_five = not_top_five[5:]
    top_five = {key:val for key, val in distances_dic.items() if val not in not_top_five}
    for key in top_five:
        top_five[key] = titles_dic[key]

    return top_five, requested.title

def clean_article(file):
    with open(file, 'r') as f:
        contents = f.read()
        contents = re.sub('\n', '', contents)
        contents = re.sub('(Mr.|Dr.|Mrs.|Jr.|Ms.|Rep.)', ' ', contents)
        contents = re.sub(r"([0-9])\.([0-9])", r"\1,\2", contents)
        contents = re.split('[\.!\?]', contents)
        contents = [x for x in contents if len(x.split(' ')) > 2]

        f.close()

    return contents

def compare_files(file1, file2):
    ar1 = clean_article(file1)
    ar2 = clean_article(file2)

    #get wmd between each sentence
    #store values in array [x, y] by index in a1, a2
    wmd_arr = []
    for i in range(len(ar1)):
        wmd_arr.append([0])
        for j in range(len(ar2) - 1):
            wmd_arr[i].append(0)

    print('Getting distances')
    for x in range(len(ar1)):
        for y in range(len(ar2)):
            wmd_arr[x][y] = get_wmd(ar1[x], ar2[y])

    #get lowest values for each sentence
    lowest = []
    for i in range(len(wmd_arr)):
        x = wmd_arr[i]
        z = ar2[x.index(min(x))]
        lowest.append([ar1[i], z, min(x)])

    return lowest

def compare_strings(str_array):
    wmds = []
    l = len(str_array)
    for i in range(l):
        wmds.append([0])
        for j in range(l - 1):
            wmds[i].append(0)

    for i in range(l):
        for j in range(l):
            wmds[i][j] = get_wmd(str_array[i], str_array[j])

    return wmds


def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', action='store', dest='url', type=str, nargs=1, help='url to the article')
    parser.add_argument('-f', action='store', dest='files', type=str, nargs=2, help='two input files')
    parser.add_argument('-s', action='store', dest='strings', type=str, nargs='*', help='two strings (wrapped in quotes)')
    args = parser.parse_args()
    if args.url:
        url = args.url[0]
        top_five, title = find_top_articles(url)
        print('Similar to \'%s\'\n' % title)
        for key in top_five:
            print('\'%s\' URL: %s\n' % (top_five[key], key))
    if args.files:
        file1 = args.files[0]
        file2 = args.files[1]
        print('Comparing %s to %s...\n' % (file1, file2))
        lowest = compare_files(file1, file2)
        print('Each sentence from %s is followed by a similar sentence from %s\n' % (file1, file2))
        for x in lowest:
            print('%s\nSimilar to:\n%s\nScore:%f' % (x[0], x[1], x[2]))

    if args.strings:
        st = args.strings
        wmds = compare_strings(st)
        print('arg i compared to arg j = [i[j]]\n')
        print(wmds)

if __name__ == "__main__":
   main(sys.argv[1:])
>>>>>>> 036606f5efa6a891bc0940ece104183b0c021e18
