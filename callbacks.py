import matplotlib.pyplot as plt
from IPython.display import clear_output
%matplotlib inline
plt.rcParams['figure.figsize'] = (20, 8)
import time

class AutoStop(keras.callbacks.Callback):
    def __init__(self, monitor='loss', value=0.004 , epoch_len=600, verbose=1):
        super(keras.callbacks.Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose
        self.epoch_len = epoch_len
        self.last_values = np.zeros(self.epoch_len)

    def on_epoch_end(self, epoch, logs={}):
        self.last_values = np.append(self.last_values[1:], [logs.get(self.monitor)])
        dt = self.value+1
        if epoch>self.epoch_len:
            dt = np.mean(np.abs(self.last_values[1:] - self.last_values[:-1]))
        if dt is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if dt < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

            
class PlotLosses(keras.callbacks.Callback):
    def __init__(self): 
        self.update_rate = 1
        self.start = 0
        self.variance = 0
    
    def on_train_begin(self, logs={}):
        self.i = self.start
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.t = time.time()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        if epoch<self.start:
            return
        
        self.logs.append(logs)
        self.x.append(epoch)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        
        t0 = time.time()
        if t0-self.t<self.update_rate:
            return
        self.t = t0
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, '-', label="loss")
        plt.plot(self.x, self.val_losses, '-', label="val_loss")
        plt.hlines(y=self.variance, xmin=0, xmax=epoch, label='variance')
        plt.yscale('log')
        plt.grid(which='both')
        plt.legend()
        plt.show();
        
        print("loss: " + str(logs.get('loss')) + " - val_loss: " + str(logs.get('val_loss')))
        