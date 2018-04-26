import sys
import os
import math
import re

class SentAnalyzer:
    class TrainSplit:
        """Represents a set of training/testing data. self.train is a list of Examples, as is self.test.
        """
        def __init__(self):
            self.train = []
            self.test = []

    class Example:
        """Represents a document with a label. klass is 'pos' or 'neg' by convention.
           words is a list of strings.
        """
        def __init__(self):
            self.klass = ''
            self.words = []

    def __init__(self):
        """SentimentAnalyzer initialization"""
        self.index_pattern= re.compile("(\d+)")
        self.phrs_pat1 = re.compile("JJ\d* NN[S]?\d* ")
        self.phrs_pat2 = re.compile("RB[S]?[R]?\d* JJ\d* (?![NN][S]?)")
        self.phrs_pat3 = re.compile("JJ\d* JJ\d* (?![NN][S]?)")
        self.phrs_pat4 = re.compile("NN[S]?\d* JJ\d* (?![NN][S]?)")
        self.phrs_pat5 = re.compile("RB[R]?[S]?\d* VB[D]?[N]?[G]?\d* ")

        self.great_count = 0.0
        self.poor_count = 0.0
        self.phrase_polarity = {}

        # stop words list from last assignment
        self.stopList = set(self.readFile('./processed_docs/english.stop'))
        self.numFolds = 10 # cross validation
        self.pos_hit = {}
        self.neg_hit = {}


    def classify(self, words):
        pos_list = []
        word_list = []
        postn = 0

        for word in words:
            splits = word.split('_')
            word_list.append(splits[0])
            pos_list.append(splits[1] + str(postn))
            postn += 1

        pos_str = ' '.join(pos_list)
        
        # for extracting patterns from pos tags
        mtchng_pat = []
        mtchng_pat.extend(self.phrs_pat1.findall(pos_str))
        mtchng_pat.extend(self.phrs_pat2.findall(pos_str))
        mtchng_pat.extend(self.phrs_pat3.findall(pos_str))
        mtchng_pat.extend(self.phrs_pat4.findall(pos_str))
        mtchng_pat.extend(self.phrs_pat5.findall(pos_str))

        polarity = 0

        for match in mtchng_pat:
            pattern_parts = match.split(' ')
            index = self.index_pattern.findall(pattern_parts[0])
            phrase_index = int(index[0])
            phrase = word_list[phrase_index] + " " + word_list[phrase_index + 1]
            polarity += self.phrase_polarity.get(phrase, 0)

        pred = 'pos' if polarity > 0 else 'neg'
        return pred

    def semanticOrient(self):
        for phrase in self.pos_hit.keys():
            # count threshold
            if self.pos_hit[phrase] < 4 and self.neg_hit[phrase] < 4:
                continue

            # calc semantic orientation
            self.phrase_polarity[phrase] = math.log(self.pos_hit[phrase] * self.poor_count, 2) - math.log(self.neg_hit[phrase] * self.great_count, 2)

    def calcNear(self, word_list, limit, i, phrase_type):
        # smoothing
        count = 0.01
        length = len(word_list)

        left_bnd = 0 if i - limit < 0 else i - limit
        right_bnd = length if i + limit + 2 > length else i + limit + 2
        phrs_end = length - 1 if i + 2 > length-1 else i + 2
        
        for j in range(left_bnd, right_bnd):
            if word_list[j] == phrase_type:
                count += 1.0
        
        # for j in range(phrs_end,right_bnd):
        #     if word_list[j] == phrase_type:
        #         count += 1.0
        
        return count

    def addExample(self, klass, words):
        word_list = []
        pos_list = []
        postn = 0
        
        for word in words:
            splits = word.split('_')
            word_list.append(splits[0])

            # great instead of excellent
            if splits[0] == "great":
                self.great_count += 1

            elif splits[0] == "poor":
                self.poor_count += 1
            
            # add pos tag with position
            pos_list.append(splits[1] + str(postn))
            postn += 1


        pos_str = ' '.join(pos_list)
        mtchng_pat = []
        mtchng_pat.extend(self.phrs_pat1.findall(pos_str))
        mtchng_pat.extend(self.phrs_pat2.findall(pos_str))
        mtchng_pat.extend(self.phrs_pat3.findall(pos_str))
        mtchng_pat.extend(self.phrs_pat4.findall(pos_str))
        mtchng_pat.extend(self.phrs_pat5.findall(pos_str))

        for match in mtchng_pat:
            pattern_parts = match.split(' ')
            index = self.index_pattern.findall(pattern_parts[0])
            phrase_index = int(index[0])
            phrs = word_list[phrase_index] + " " + word_list[phrase_index + 1]

            # print (phrs)
            self.pos_hit[phrs] = self.pos_hit.get(phrs, 0.0) + self.calcNear(word_list, 10, phrase_index, "great")
            self.neg_hit[phrs] = self.neg_hit.get(phrs, 0.0) + self.calcNear(word_list, 10, phrase_index, "poor")




    # END TODO (Modify code beyond here with caution)
    #############################################################################


    def readFile(self, fileName):
        """
         * Code for reading a file.  you probably don't want to modify anything here,
         * unless you don't like the way we segment files.
        """
        contents = []
        f = open(fileName)
        for line in f:
            contents.append(line)
        f.close()
        result = self.segmentWords('\n'.join(contents))
        return result

    def segmentWords(self, s):
        """
         * Splits lines on whitespace for file reading
        """
        return s.split()

    def trainSplit(self, trainDir):
        """Takes in a trainDir, returns one TrainSplit with train set."""
        split = self.TrainSplit()
        posTrainFileNames = os.listdir('%s/pos/' % trainDir)
        negTrainFileNames = os.listdir('%s/neg/' % trainDir)
        for fileName in posTrainFileNames:
            example = self.Example()
            example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
            example.klass = 'pos'
            split.train.append(example)
        for fileName in negTrainFileNames:
            example = self.Example()
            example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
            example.klass = 'neg'
            split.train.append(example)
        return split

    def train(self, split):
        for example in split.train:
            words = example.words
            self.addExample(example.klass, words)

    def crossValidationSplits(self, trainDir):
        """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
        splits = []
        posTrainFileNames = os.listdir('%s/pos/' % trainDir)
        negTrainFileNames = os.listdir('%s/neg/' % trainDir)
        # for fileName in trainFileNames:
        for fold in range(0, self.numFolds):
            split = self.TrainSplit()
            for fileName in posTrainFileNames:
                example = self.Example()
                example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
                example.klass = 'pos'
                if fileName[2] == str(fold):
                    split.test.append(example)
                else:
                    split.train.append(example)
            for fileName in negTrainFileNames:
                example = self.Example()
                example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
                example.klass = 'neg'
                if fileName[2] == str(fold):
                    split.test.append(example)
                else:
                    split.train.append(example)
            splits.append(split)
        return splits

    def filterStopWords(self, words):
        """Filters stop words."""
        filtered = []
        for word in words:
            if not word in self.stopList and word.strip() != '':
                filtered.append(word)
        return filtered


def test10Fold(args):
    nb = SentAnalyzer()
    splits = nb.crossValidationSplits(args[0])
    avgAccuracy = 0.0
    fold = 0
    for split in splits:
        classifier = SentAnalyzer()
        accuracy = 0.0
        for example in split.train:
            words = example.words
            classifier.addExample(example.klass, words)

        classifier.semanticOrient()

        for example in split.test:
            words = example.words
            guess = classifier.classify(words)
            if example.klass == guess:
                accuracy += 1.0

        #print(accuracy, len(split.test))

        accuracy = accuracy / len(split.test)
        avgAccuracy += accuracy
        print '[INFO]\tFold %d Accuracy: %f' % (fold, accuracy)
        fold += 1
    avgAccuracy = avgAccuracy / fold
    print '[INFO]\tAccuracy: %f' % avgAccuracy


def classifyDir(trainDir, testDir):
    classifier = SentAnalyzer()
    trainSplit = classifier.trainSplit(trainDir)
    classifier.train(trainSplit)
    testSplit = classifier.trainSplit(testDir)
    accuracy = 0.0

    classifier.semanticOrient()

    for example in testSplit.train:
        words = example.words
        guess = classifier.classify(words)
        if example.klass == guess:
            accuracy += 1.0
    accuracy = accuracy / len(testSplit.train)
    print '[INFO]\tAccuracy: %f' % accuracy


def main():
    args=sys.argv[1:]

    if len(args) == 2:
        classifyDir( args[0], args[1])
    elif len(args) == 1:
        test10Fold(args)


if __name__ == "__main__":
    main()
