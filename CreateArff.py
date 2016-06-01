# script to create a dummy arff file from numpy features and labels

def CreateArff(file_name, features, labels=[]):

   dim_features = features.shape[1]
   count_data = features.shape[0]
   if len(labels) == 0: labels = ['?' for i in range(count_data)]

   f1 = open(file_name, 'w')
   
   # creating header
   f1.write('@relation ComParE_2013_Acoustic_Features\n\n')
   f1.write('@attribute name string\n')

   for i in range(dim_features):
      to_write = '@attribute feature' + str(i) + ' numeric\n'
      f1.write(to_write)

   f1.write('@attribute sinc numeric\n')
   f1.write('@data\n')
   
   for i in range(count_data):
      cur_features = ','.join(map(str,list(features[i,:])))
      cur_label = str(labels[i])
      to_write = 'file_id' + str(i) + ',' + cur_features + ',' + cur_label + '\n'
      f1.write(to_write)

   f1.close()

      
