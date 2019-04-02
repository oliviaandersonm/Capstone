#wmd_model.py
#get vectorizing model - use google news training
import os
from gensim.models import KeyedVectors
<<<<<<< HEAD
print('Beginning to load vectors')
model = KeyedVectors.load_word2vec_format(os.path.expanduser('~/Capstone/data/wmd/GoogleNews-vectors-negative300.bin.gz'), binary=True)
model.init_sims(replace=True)
print('Vectors loaded')
#save
print('Saving vectors')
model.wv.save_word2vec_format(os.path.expanduser('~/Capstone/data/wmd/google_vectors.txt'), binary=False)
print('Vectors saved to data/wmd/google_vectors.txt')
=======
model = KeyedVectors.load_word2vec_format(os.path.expanduser('~/pythonml/CapData/GoogleNews-vectors-negative300.bin.gz'), binary=True)
model.init_sims(replace=True)

#save
model.wv.save('wmd_vectors.kv')
>>>>>>> 036606f5efa6a891bc0940ece104183b0c021e18
