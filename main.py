import tqdm
import gzip
import time
import numpy as np
from XStream_River import xStream
from rs_hash import RSHash
from loda import LODA
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.datasets import load_svmlight_file
import warnings
warnings.filterwarnings("ignore")


#Select algorithm
algorithm = None
while True:
  print("Select algorithm:")
  print("1. XStream")
  print("2. Loda")
  print("3. RSHash")
  choice = input("Enter your choice: ")

  try:
    algorithm = int(choice)
    if algorithm < 1 or algorithm > 3:
      print("Your choice must be 1, 2 or 3\n")
    else:
      break
  except:
    print("Input must be a number!\n")
  
#Select dataset
dataset = None
while True:
  print("Select dataset:")
  print("1. cancer")
  print("2. ionosphere")
  print("3. telescope")
  print("4. indians")
  print("5. gisette")
  print("6. isolet")
  print("7. letter")
  print("8. madelon")
  print("9. spam-sms")
  choice = input("Enter your choice: ")

  try:
    dataset = int(choice)
    if dataset < 1 or dataset > 9:
      print("Your choice must be between 1 and 9.\n")
    else:
      break
  except:
    print("Input must be a number!\n")
  
all_datasets = ['cancer', 'ionosphere', 'telescope', 'indians', 'gisette', 'isolet', 'letter', 'madelon', 'spam-sms']

#Read dataset
if dataset == 9:
  data = load_svmlight_file("data/Row-streaming/spam-sms")
else:
  data = gzip.open("data/Static/" + all_datasets[dataset-1] + ".gz", "r")

print("Dataset: ", all_datasets[dataset-1])

if dataset < 9: #Not spam-sms
  X, y = [], []
  for i in data:
    i = (i.decode('utf-8')).split(",")
    i = [float(x) for x in i]
    X.append(np.array(i[:-1]))
    y.append(i[-1])
else:
  X = data[0]
  y = data[1]
  X= X.todense()

#Create models
window_size = int(0.25*len(y))
k = 100
n_chains = 100
depth = 15

if algorithm == 1:
  cf = xStream(num_components=k, n_chains=n_chains, depth=depth, window_size=window_size) 
elif algorithm ==2:
  cf = LODA(num_bins=100, num_random_cuts=100) 
else:
  if dataset == 9:
    cf = RSHash(X.min(axis=0).A1, X.max(axis=0).A1)
  else:
    X = np.array(X)
    cf = RSHash(X.min(axis=0), X.max(axis=0))


all_scores = []
start_time = time.time()

for i, sample in enumerate(tqdm.tqdm(X)):
  if dataset == 9: #spam-sms
    cf.learn_one(sample.A1)
  else:
    cf.learn_one(sample)

  if i>=window_size:
    if dataset == 9: #spam-sms
      anomalyscore = -cf.predict_one(sample.A1)
    else:
      anomalyscore = -cf.predict_one(sample)

    if algorithm == 3:
      all_scores.append(anomalyscore)
    else:
       all_scores.append(anomalyscore[0])

print('Time: %.3f seconds' % (time.time() - start_time))

y_adjusted = y[window_size:window_size+len(all_scores)]

chunks = [all_scores[x:x+window_size] for x in range(0, len(all_scores), window_size)]
y_chunks = [y_adjusted[x:x+window_size] for x in range(0, len(y_adjusted), window_size)]

AP_window = []

for i in range(len(y_chunks)-1):
  score = average_precision_score(y_chunks[i], chunks[i])
  AP_window.append(score)

OAP = average_precision_score(y_adjusted, all_scores) 
MAP = sum(AP_window)/len(AP_window)
AUC = roc_auc_score(y_adjusted, all_scores)

if algorithm == 1:
  print("XStream results:")
elif algorithm ==2:
  print("Loda results: ")
else:
  print("RSHash: ")

print("\tOAP =", OAP,"\n",
      "\tMAP =", MAP, "\n", 
      "\tAUC =", AUC)