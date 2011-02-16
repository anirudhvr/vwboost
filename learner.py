# an abstract class for making the boosting impl. more generic

import sys

class learner:
    def __init__(self):
        pass


    def train_file(self, filename, trainopts = {}):
        sys.stderr.write("Not implemented in abstract class\n")
        #pass

    def train_line(self, line, trainopts = {}, weight = 1.0):
        sys.stderr.write("Not implemented in abstract class\n")
        #pass

    def test_file(self, filename, testopts = {}, output_to_file=None):
        sys.stderr.write("Not implemented in abstract class\n")
        #pass

    def test_line(self, line, testopts = {}):
        sys.stderr.write("Not implemented in abstract class\n")
        #pass




