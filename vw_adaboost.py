import sys, os, re, math
import subprocess, logging, random
import learner
import vwpy
from vwpy import *;
from math import exp;     
from itertools import izip;

logging.basicConfig(level=logging.DEBUG,stream=sys.stderr,
                    format='%(asctime)s %(name)s: %(levelname)s\
                    %(message)s')
log = logging.getLogger('vw_adaboost')

class vwboost:
    def __init__(self, training_file, trainopts = {}, 
                 rounds = 10, debug = True):
        if not os.access(training_file, os.R_OK):
            raise StandardError, "Cannot read training file"

        # count number of examples as the number of lines in the file
        self.num_examples = 0.0
        for l in open(training_file, "r"):
            self.num_examples += 1.0

        self.debug = debug
        self.num_rounds = rounds
        self.current_round = 1

        self.learners = list()
        self.alphas = list()
        
        self.training_file = training_file
        self.trainopts = trainopts
        self.vws = list()
        self.is_classifying = False



    def boost(self, threshold = 0.0, max_error = 0.5):
         
        def sign(x):          
            # takes a number and returns +1,-1 depending on sign
            return((x>0)*2-1);

        int_training_file = "%s_new" %(self.training_file)                                                                                                  
        # read in labels
        labels=[float(l.split(' ',1)[0])==1.0 for l in open(self.training_file, "r")];                         
        ninputs=len(labels);        
        weights=[1.0/ninputs]*ninputs;
        preds=[0.0]*ninputs;
                                    
        self.alphas=[];
        self.learners=[];
        total=[0.0]*ninputs;
        log.info("Starting boosting for %d rounds" %self.num_rounds)
                           
              

        #while error < max_error and round < self.num_rounds:
        while self.current_round <= self.num_rounds:
            # Step 1
            # Train weak learner using distribution
            log.info("Training in round %d" %self.current_round)
            # Do learning here, from rounds 2 ... 
            vw_train = vwpy()
            trainopts = self.trainopts                                             
            passes = trainopts['passes']                                                       

            #if 'final_regressor' not in trainopts:
            trainopts['final_regressor'] = "round_%d_regressor" %self.current_round
                
            # XXX we can use train_file only because the weight modifier is 1
            trainopts['passes'] = passes                                
                
            f=open(int_training_file,'w');
            for (i,l) in enumerate(open(self.training_file)):
                [header,features]=l.split('|',1);
                f.write("%i %e |%s" % (labels[i],weights[i]*ninputs,features))
            f.close();
                    
            os.system('rm '+int_training_file+'.cache');
            vw_train.train_file(int_training_file, trainopts)
            vw_train.train_end()

            if self.debug:
                    log.info("Training done in round %d" %self.current_round)
                    log.info("Classifier at %s"
                             %trainopts['final_regressor'])


            # Step 2
            # use the last learner as weak hypothesis h_t
            # and find the error rate for this round
            log.info("Testing in round %d" %self.current_round)
            
            # get predictions
            vw_test = vwpy()                   
            preds=map(lambda a:sign(float(a)*2.-1.),vw_test.run_vw(self.training_file,{'cache':' ','initial_regressor':trainopts['final_regressor'],'testonly': ' '}))
            vw_test.test_end();                 
                             
            # compute correct and wrong examples
            error = reduce(lambda s,i: s+((preds[i]>=threshold)!=labels[i])*weights[i],xrange(ninputs),0.0);                  

            # Step 3
            # Compute error, alpha
            alpha = 0.5 * math.log((1.0 - error)/error)
                                                       
            # udpate training error
            for i in xrange(len(total)):
                total[i]+=alpha*preds[i];
            gcorr=reduce(lambda s,i: s+((total[i]>=threshold)==labels[i]),xrange(ninputs),0.0);           

            log.info("Round %d, error: %f, alpha: %f, Tr-Accuracy:%2.4f" %(self.current_round, error, alpha,gcorr/float(ninputs)))

            if error>=.5: 
                log.info('Weak learner too weak ... stopping.');
                break; # quit - weak learner officially sucks
            self.alphas.append(alpha) # store for use in the final classifier
            self.learners.append(trainopts['final_regressor'])


            # Step 4
            # Update weights of training examples using alpha
            pos=math.exp(-alpha);
            neg=math.exp( alpha);
            for i in xrange(ninputs):
                if (preds[i]>=threshold) == labels[i]: 
                    weights[i]*=pos;   # decrease weight
                else: 
                    weights[i]*=neg;   # increase weight
            
            # normalize weights;
            Z=sum(weights);
            weights=map(lambda a: a/Z,weights); 

            # first read the _tmp file in
            self.current_round += 1 
            


    # Note: this thing is a generator
    def classify(self, testing_file):        
        if not self.is_classifying:
            self.num_rounds=len(self.learners); # could have stopped prematurely
            assert len(self.learners) == len(self.alphas)
            r = self.num_rounds
            while r > 0:
                self.vws.append(vwpy())
                r -= 1

        # start testing
        for line in open(testing_file):
            score = 0.0
            for i in range(self.num_rounds):
                res = self.vws[i].test_line(line, {'initial_regressor': self.learners[i]})*2.0-1.0;
                score += float(res>=0) * self.alphas[i]
            #yield(score/self.num_rounds)
            yield((score+1.0)/2.0)

    

if __name__ == "__main__":
    boost = vwboost("train.dat", {'passes':20, 'cache':'','learning_rate':10,                                   
                                  'predictions':'/dev/null','bit_precision':6}, rounds=10)
    #boost = vwboost("train_tmp.txt", {'passes':10}, rounds=50)
    #boost.train_initial_classifier()
    boost.boost()
    for i in boost.classify("test.dat"): 
        print i

