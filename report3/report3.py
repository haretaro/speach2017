import chainer
from chainer.datasets import tuple_dataset
import chainer.functions as F
import chainer.links as L
from chainer import reporter
from chainer import training
from chainer.training import extensions
import numpy as np
import matplotlib.pyplot as plt
import os

batchsize = 16
gpu = -1
out = 'result'
epoch = 1
n_units = 100

train_src = np.loadtxt('train_source.csv', dtype=np.float32, delimiter=',')
train_target = np.loadtxt('train_target.csv', dtype=np.float32, delimiter=',')
test_src = np.loadtxt('test_source.csv', dtype=np.float32, delimiter=',')
test_target = np.loadtxt('test_target.csv', dtype=np.float32, delimiter=',')

train = tuple_dataset.TupleDataset(train_src, train_target)
test = tuple_dataset.TupleDataset(test_src, test_target)

class NN(chainer.Chain):

    def __init__(self, n_units):
        super(NN, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_units)
            self.l2 = L.Linear(None, n_units)
            self.l3 = L.Linear(None, 3)

    def predict(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

    def __call__(self, x, t):
        self.loss= F.mean_squared_error(self.predict(x), t)
        reporter.report({'loss': self.loss}, self)
        return self.loss

model = NN(n_units)
optimizer = chainer.optimizers.Adam()
optimizer.setup(model)

train_iter = chainer.iterators.SerialIterator(train, batchsize)
test_iter = chainer.iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)

updater = training.StandardUpdater(train_iter, optimizer, device=gpu)
trainer = training.Trainer(updater, (epoch, 'epoch'), out=out)

trainer.extend(extensions.Evaluator(test_iter, model, device=gpu))
trainer.extend(extensions.LogReport())

if extensions.PlotReport.available():
    trainer.extend(
        extensions.PlotReport(['main/loss', 'validation/main/loss'],
                              'epoch', file_name='loss.png'))

trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss', 'elapsed_time']))

trainer.extend(extensions.ProgressBar())

trainer.run()

pred = model.predict(test_src).data

for d in range(3):
    plt.figure()
    plt.plot(pred[:,d], 'or', label='変換データ', markersize=4, alpha=1)
    plt.plot(test_target[:,d], 'og', label='正解データ', markersize=2, alpha=1)
    plt.xlabel('時間 [フレーム]')
    plt.ylabel('ケプストラム')
    plt.title('{}次元目のケプストラム'.format(d))
    plt.legend()
    plt.savefig(os.path.join(out, 'fig{}'.format(d)))
plt.show()
