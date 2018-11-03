from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# load data.
root='model_fold4_1531419430/'
embeddings_train_set = np.load(root+'embeddings_train_set.npz')
gt_train_set = np.load(root+'gt_train_set.npz')
embeddings_test_set = np.load(root+'embeddings_test_set.npz')
gt_test_set = np.load(root+'gt_test_set.npz')
protos = np.load(root+'prototypes.npz')

# prepare train data.
classes_train = gt_train_set['arr_0']
labels_train = [np.int(np.squeeze(np.where(np.array(classes_train[i]) == max(classes_train[i])))) for i in range(classes_train.shape[0])]
emb_train = embeddings_train_set['arr_0']

# prepare test data.
PERCENTAGE_DATA = 0.02
ammount_data = embeddings_test_set['arr_0'].shape[0]
selected_ids = np.random.randint(0,ammount_data,int(ammount_data*PERCENTAGE_DATA))
classes_test = gt_test_set['arr_0']
labels = [np.int(np.squeeze(np.where(np.array(classes_test[i])==max(classes_test[i])))) for i in selected_ids] + labels_train + list(range(10))
emb = np.concatenate((embeddings_test_set['arr_0'][selected_ids], emb_train, protos['arr_0']), axis=0)

# compute t-SNE or PCA.
PERPLEX=30.0
X_tsne = TSNE(learning_rate=100,perplexity=PERPLEX, init='pca').fit_transform(emb)
X_pca = PCA().fit_transform(emb)

# plots.
plt.subplot(221)
plt.title('t-SNE: Test data')
plt.scatter(X_tsne[:len(selected_ids), 0], X_tsne[:len(selected_ids), 1], c=labels[:len(selected_ids)], cmap='inferno', alpha=0.2)
plt.scatter(X_tsne[-10, 0], X_tsne[-10, 1], c='r', alpha=1, marker='d', label='car_horn')
plt.scatter(X_tsne[-9, 0], X_tsne[-9, 1], c='r', alpha=1, marker=',', label='children_playing')
plt.scatter(X_tsne[-8, 0], X_tsne[-8, 1], c='r', alpha=1, marker='o', label='dog_bark')
plt.scatter(X_tsne[-7, 0], X_tsne[-7, 1], c='r', alpha=1, marker='v', label='drilling')
plt.scatter(X_tsne[-6, 0], X_tsne[-6, 1], c='r', alpha=1, marker='^', label='engine_idling')
plt.scatter(X_tsne[-5, 0], X_tsne[-5, 1], c='r', alpha=1, marker='>', label='gun_shot')
plt.scatter(X_tsne[-4, 0], X_tsne[-4, 1], c='r', alpha=1, marker='<', label='jackhammer')
plt.scatter(X_tsne[-3, 0], X_tsne[-3, 1], c='r', alpha=1, marker='x', label='siren')
plt.scatter(X_tsne[-2, 0], X_tsne[-2, 1], c='r', alpha=1, marker='+', label='street_music')
plt.scatter(X_tsne[-1, 0], X_tsne[-1, 1], c='r', alpha=1, marker='*', label='air_conditioner')
plt.legend(bbox_to_anchor=(0., 1.25, 2.2, .102), ncol=5, mode="expand")

plt.subplot(222)
plt.title('PCA: Test data')
plt.scatter(X_pca[:len(selected_ids), 0], X_pca[:len(selected_ids), 1], c=labels[:len(selected_ids)], cmap='inferno', alpha=0.2)
plt.scatter(X_pca[-10, 0], X_pca[-10, 1], c='r', alpha=1, marker='d')
plt.scatter(X_pca[-9, 0], X_pca[-9, 1], c='r', alpha=1, marker=',')
plt.scatter(X_pca[-8, 0], X_pca[-8, 1], c='r', alpha=1, marker='o')
plt.scatter(X_pca[-7, 0], X_pca[-7, 1], c='r', alpha=1, marker='v')
plt.scatter(X_pca[-6, 0], X_pca[-6, 1], c='r', alpha=1, marker='^')
plt.scatter(X_pca[-5, 0], X_pca[-5, 1], c='r', alpha=1, marker='>')
plt.scatter(X_pca[-4, 0], X_pca[-4, 1], c='r', alpha=1, marker='<')
plt.scatter(X_pca[-3, 0], X_pca[-3, 1], c='r', alpha=1, marker='x')
plt.scatter(X_pca[-2, 0], X_pca[-2, 1], c='r', alpha=1, marker='+')
plt.scatter(X_pca[-1, 0], X_pca[-1, 1], c='r', alpha=1, marker='*')

plt.subplot(223)
plt.title('t-SNE: Train data')
plt.scatter(X_tsne[len(selected_ids):-10, 0], X_tsne[len(selected_ids):-10, 1], c=labels[len(selected_ids):-10], cmap='inferno', alpha=0.2)
plt.scatter(X_tsne[-10, 0], X_tsne[-10, 1], c='r', alpha=1, marker='d')
plt.scatter(X_tsne[-9, 0], X_tsne[-9, 1], c='r', alpha=1, marker=',')
plt.scatter(X_tsne[-8, 0], X_tsne[-8, 1], c='r', alpha=1, marker='o')
plt.scatter(X_tsne[-7, 0], X_tsne[-7, 1], c='r', alpha=1, marker='v')
plt.scatter(X_tsne[-6, 0], X_tsne[-6, 1], c='r', alpha=1, marker='^')
plt.scatter(X_tsne[-5, 0], X_tsne[-5, 1], c='r', alpha=1, marker='>')
plt.scatter(X_tsne[-4, 0], X_tsne[-4, 1], c='r', alpha=1, marker='<')
plt.scatter(X_tsne[-3, 0], X_tsne[-3, 1], c='r', alpha=1, marker='x')
plt.scatter(X_tsne[-2, 0], X_tsne[-2, 1], c='r', alpha=1, marker='+')
plt.scatter(X_tsne[-1, 0], X_tsne[-1, 1], c='r', alpha=1, marker='*')

plt.subplot(224)
plt.title('PCA: Train data')
plt.scatter(X_pca[len(selected_ids):-10, 0], X_pca[len(selected_ids):-10, 1], c=labels[len(selected_ids):-10], cmap='inferno', alpha=0.2)
plt.scatter(X_pca[-10, 0], X_pca[-10, 1], c='r', alpha=1, marker='d')
plt.scatter(X_pca[-9, 0], X_pca[-9, 1], c='r', alpha=1, marker=',')
plt.scatter(X_pca[-8, 0], X_pca[-8, 1], c='r', alpha=1, marker='o')
plt.scatter(X_pca[-7, 0], X_pca[-7, 1], c='r', alpha=1, marker='v')
plt.scatter(X_pca[-6, 0], X_pca[-6, 1], c='r', alpha=1, marker='^')
plt.scatter(X_pca[-5, 0], X_pca[-5, 1], c='r', alpha=1, marker='>')
plt.scatter(X_pca[-4, 0], X_pca[-4, 1], c='r', alpha=1, marker='<')
plt.scatter(X_pca[-3, 0], X_pca[-3, 1], c='r', alpha=1, marker='x')
plt.scatter(X_pca[-2, 0], X_pca[-2, 1], c='r', alpha=1, marker='+')
plt.scatter(X_pca[-1, 0], X_pca[-1, 1], c='r', alpha=1, marker='*')
plt.show()

# class correspondences, for the record.
classes = {}
classes['0'] = 'car_horn'
classes['1'] = 'children_playing'
classes['2'] = 'dog_bark'
classes['3'] = 'drilling'
classes['4'] = 'engine_idling'
classes['5'] = 'gun_shot'
classes['6'] = 'jackhammer'
classes['7'] = 'siren'
classes['8'] = 'street_music'
classes['9'] = 'air_conditioner'
