import sys, os, re, math
import subprocess, logging, random
from math import exp
import pdb
from learner import *; # for abstract class representation
import time

logging.basicConfig(level=logging.DEBUG,stream=sys.stderr,
                    format='%(asctime)s %(name)s: %(levelname)s\
                    %(message)s')
log = logging.getLogger('vwpy')

class vwpy(learner):
    #vw_commonopts = {'bit_precision':24, 'threads':1,
    #'Loss':1,'learning_rate':1280000, 'power_t':1,
    #'initial_t':128000, 'exact': ''}
    vw_commonopts = {'threads':1, 'exact': ''}
    vw_trainopts = {'passes':1, 'final_regressor': 'r_temp',
                    'predictions':'/dev/stdout',
                    'bit_precision':24, 'Loss':1};
#                    'learning_rate':1, 'power_t':1,
#                    'initial_t':1 }
    vw_testopts = {'testonly':'', 'initial_regressor':'r_temp',
                   'predictions':'/dev/stdout',
                   'bit_precision':24, 'Loss':1}
#                   'learning_rate':1, 'power_t':1,
#                   'initial_t':1 }

    vwpath = "~/bin/vw"
    debug = True
    training_done = False
    testing_done = False

    def __init__(self, vwpath = '', debug = True, 
               vw_commonopts = {}):
      if vwpath != '' and os.access(vwpath, os.X_OK):
          self.vwpath = vwpath

      self.debug = debug
      self.stderr = open("vwstderr.txt", "w")
      self.stdout = open("vwstdout.txt", "w")
      #self.stdout = sys.stdout

      for k in vw_commonopts.iterkeys():
          self.vw_commonopts[k] = vw_commonopts[k]

      vwflags = map(lambda x: "--%s %s" %(x, self.vw_commonopts[x]),
                    self.vw_commonopts.keys() )
      self.vwcommand = "%s %s" %( self.vwpath, ' '.join(vwflags))

      if self.debug:
          log.info("VW basic command: %s" %self.vwcommand) 

      self.training  = False
      self.testing   = False

    def create_pipe(self, options, out = None, err = None):
      if out == None: out = self.stdout
      if err == None: err = self.stderr
      extraopts = ' '.join(map(lambda x: "--%s %s" %(x, options[x]),
                               options.keys()))
      command = self.vwcommand + " " + extraopts
      pipe = subprocess.Popen(command, shell=True, bufsize=1,
                              stdout=out,
                              stderr=err,
                              stdin=subprocess.PIPE)
      if self.debug:
          log.info("VW pipe created using command: %s" %command) 

      return pipe

    def run_vw(self, filename, options, out = None, err = None):
        if out == None: out = self.stdout ## XXX ignored for now
        if err == None: err = self.stderr ## XXX ignored for now
        extraopts = ' '.join(map(lambda x: "--%s %s" %(x, options[x]),options.keys()))
        tmpfile='/tmp/vwoutput_'+str(random.randint(0,9999999))+'.csv';
        command = self.vwcommand + " -d "+filename+" " + extraopts +' -p '+tmpfile
        log.info('RUN_VW: %s'% command);
        os.system(command);
        if self.debug: log.info("VW pipe created using command: %s" %command) 
        output=[x for x in open(tmpfile).read().split('\n') if len(x)>0];
        os.system('rm '+tmpfile);
        return(output);
        



    def train_start(self, vw_trainopts = {}):
        # create a training instance if we're not already training
        if not self.training:
            for k in self.vw_trainopts.iterkeys():
                if k not in vw_trainopts:
                    vw_trainopts[k] = self.vw_trainopts[k]

        # let stdout and stderr of pipe go to self.whatever
        self.train_pipe = self.create_pipe(vw_trainopts) 
        if self.debug:
            log.info("VW training started")
            #log.info("VW training started...message=%s")
            #%('\n'.join([self.train_pipe.stdout.readline(),
            #self.train_pipe.stderr.readline()])))
            self.training = True
            self.training_done = True
        else:
            if self.debug:
                log.info("Already training. Not doing anything!\n")
                return False

        return True

    def train_file_linebyline(self, inputfile, vw_trainopts = {}):
        passes = self.vw_trainopts['passes']
        if 'passes' in vw_trainopts: 
            passes = vw_trainopts['passes']

        vw_trainopts['data'] = inputfile

        if not self.training:
            if not self.train_start(vw_trainopts):
                return False

        file = open(inputfile)
        while passes > 0:
            file.seek(0)
            passes -= 1
            for line in file:
                self.train_pipe.stdin.write("%s" %line) # newline as well, which is good

        file.close()

        if self.debug:
            log.info("Done writing lines")
            log.info("VW training finished")
            log.info("Final regressor at %s" %vw_trainopts['final_regressor'])

        #self.train_pipe.close()
        #self.train_pipe.terminate()
        self.train_end() # needed!
        return True

    def train_file(self, inputfile, vw_trainopts = {}):
        passes = self.vw_trainopts['passes']
        if 'passes' in vw_trainopts: 
            passes = vw_trainopts['passes']
        else:
            vw_trainopts['passes'] = self.vw_trainopts['passes']

        vw_trainopts['data'] = inputfile

        if not self.training:
            if not self.train_start(vw_trainopts):
                return False

        self.train_pipe.wait()

        if self.debug:
            log.info("Done writing lines")
            log.info("VW training finished")
            log.info("Final regressor at %s" %vw_trainopts['final_regressor'])

        #self.train_pipe.close()
        #self.train_pipe.terminate()
        self.train_end() # needed!
        return True


    def train_line(self, line, vw_trainopts = {}, weight_modifier = 1.0):
        if not self.training:
            if not self.train_start(vw_trainopts):
                return False

        # we want to adjust the existing weight on this example
        if weight_modifier != 1.0:
            wt = 0.0
            fields = re.split(r'\s+', line)
            try:
                wt = float(fields[1])
            except ValueError:
                return False # indicates there was a problem re-weighting

            # by default, the 'weight' argument is multiplied by the existing wt
            fields[1] = "%.4f" %(wt * weight_modifier)
            line = ' '.join(fields)

        self.train_pipe.stdin.write(line) # XXX: assumes \n
        return True

    def train_end(self):
        if self.training:
            self.train_pipe.stdin.close()
            self.training = False
        time.sleep(0.5)


    def test_start(self, vw_testopts = {}, output_to_file = None):
        for k in self.vw_testopts.iterkeys():
            if k not in vw_testopts:
                vw_testopts[k] = self.vw_testopts[k]

        try: 
            if output_to_file != None:
                # XXX: catch exception here?
                self.test_output = open(output_to_file, "w")
                self.test_pipe = self.create_pipe(vw_testopts, self.test_output)
            else:
                self.test_pipe = self.create_pipe(vw_testopts, subprocess.PIPE)

            if self.debug:
                log.info("VW testing started")

            self.testing = True
            self.testing_done = True
        except:
            print "Exception"
            return False

        return True



    def test_file(self, inputfile, vw_testopts = {}, output_to_file=None):
        if not self.testing:
            if not self.test_start(vw_testopts, output_to_file):
                print "Error"
                 
                   
        for line in open(inputfile):
            if line=='': break;
            print line;
            self.test_pipe.stdin.write(line) # newline as well, which is good
            if output_to_file == None:
                result = self.test_pipe.stdout.readline()
                yield result

        self.test_pipe.stdin.flush()
        self.test_pipe.terminate()
        self.testing = False
        if output_to_file != None: 
            #self.test_pipe.stdout.flush()
            self.test_output.flush()
            self.test_output.close()
        if self.debug:
            log.info("VW testing finished")
            log.info("Final classifications at %s\n" %output_to_file)

    def test_line(self, line, vw_testopts = {}):
        if not self.testing:
            if not self.test_start(vw_testopts):
                return False

        self.test_pipe.stdin.write("%s" %line) # newline as well, which is good
        result = self.test_pipe.stdout.readline()
        res_num = re.match("\s*(\d+(\.\d+)?)\s*", result).group(1)
        return float(res_num.rstrip())

    def test_end(self):
        if self.testing:
            self.test_pipe.stdin.close()
            self.test_pipe.stdout.close()
            self.testing  = False

    def __del__(self):
        #self.train_pipe.stdin.close()
        if self.training_done:
            #self.train_pipe.wait()
            #self.train_pipe.wait()
            if self.training:
                self.train_end()
            pass
        if self.testing_done:
            if self.testing:
                self.test_end()
            pass
            #self.test_pipe.wait()
        sys.stderr.write("Destroying vwpy object\n")
        self.stderr.close()
        self.stdout.close()


if __name__=="__main__":
    vw = vwpy()
    if not vw.train_file("train.dat", {'passes':10,
                                       'final_regressor':'myregressor'}):
        print "Could not train!"
    while vw.training: pass
    # This is damn weird, but python sometimes doesnt write
    # the output regressor file without a little sleeping :\
    #pdb.set_trace()
    vw2 = vwpy()
    for l in open("test.dat"):
        print vw2.test_line(l, {'initial_regressor':'myregressor'}) 
    vw2.test_end()

