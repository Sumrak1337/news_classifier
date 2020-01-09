from classifier import Classifier
import datetime
import codecs
import sys

DUMP_DIR = 'dump_60k'
CLASSIFY_DATA = 'news_test.txt'
OUTPUT_DATA = 'output.txt'


if __name__ == '__main__':
    naive_bayes = Classifier()

    fileObj = codecs.open(CLASSIFY_DATA, 'r', 'utf_8_sig')
    content = fileObj.readlines()
    fileObj.close()

    naive_bayes.load(DUMP_DIR)

    fileObj = codecs.open(OUTPUT_DATA, 'w+', 'utf_8_sig')
    n = len(content)
    start = datetime.datetime.now()

    for (i, line) in enumerate(content):
        predict = naive_bayes.predict(line)
        fileObj.write('%s\r\n' % predict)

        if (i + 1) % 100 == 0:
            sys.stdout.write('\r' + '{} / {} ( {}% ) | {}s ( {}m )'.format(i + 1, n, int((i + 1) / n * 100), (datetime.datetime.now() - start).seconds, (datetime.datetime.now() - start).seconds // 60))

    fileObj.close()