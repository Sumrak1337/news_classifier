from classifier import Classifier
import datetime
import codecs
import sys

DUMP_DIR = 'dump_60k'
TRAIN_DATA = 'news_train.txt'


if __name__ == '__main__':
    naive_bayes = Classifier()

    fileObj = codecs.open(TRAIN_DATA, 'r', 'utf_8_sig')
    content = fileObj.readlines()
    fileObj.close()

    start = datetime.datetime.now()
    n = len(content)

    for (i, line) in enumerate(content):
        category, title, description = line.split('\t')
        naive_bayes.train(description, category)

        if (i + 1) % 10 == 0:
            sys.stdout.write('\r' + '{} / {} ( {}% ) | {}s ( {}m )'.format(i + 1, n, int((i + 1) / n * 100), (datetime.datetime.now() - start).seconds, (datetime.datetime.now() - start).seconds // 60))

    naive_bayes.save(DUMP_DIR)