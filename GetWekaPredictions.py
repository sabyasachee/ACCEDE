import re
import numpy

def GetWekaPredictions(input_file):

   with open(input_file) as f_ip:
      lines = f_ip.readlines()
      start_appending, predictions = 0, []
      for line in lines:
         if line == '\n': continue
         if start_appending == 1:
            line = line.strip('\n')
            line = re.sub(' +', ' ', line)
            cur_prediction = line.split(' ')[3] 
            predictions.append(cur_prediction)
         if 'predicted' in line: start_appending = 1
   
   predictions = numpy.asarray(predictions)
 
   return predictions[0:]
