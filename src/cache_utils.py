import numpy as np




class CacheEpoch:
    """ Cache within an epoch.
    """
    def __init__(self, tag):
        """ Initialize cache attributes.

        Args:
            tag (str): "train" or "test"
        """
        self.tag = tag

        self._cnt = 0
        self._cum_mse = 0
        self._cum_rms = 0

        self.avg_mse = 0
        self.avg_rms = 0

    def update_cache(self, pred_mu, target):
        """ Update cache for a learning update of a batch to the
            cummulative attributes.

        Args:
            pred_mu (torch.Tensor): predictions of the batch samples
            target (torch.Tensor): targets of the batch samples
        """
        err = (pred_mu - target).cpu().detach().numpy()
        err = err * err  # squared error
        self._cum_mse += err.mean()
        self._cum_rms += err.std()
        self._cnt += 1

    def calc_avg_across_batches(self):
        """ Calculate the average of every attribute within a complete
            epoch (accross all batches).
        """
        self.avg_mse = self._cum_mse / self._cnt
        self.avg_rms = self._cum_rms / self._cnt

    def print_cache(self, epoch):
        """ Print cache attributes (average over epoch).

        Args:
            epoch (int): epoch
        """
        print(f"\n{self.tag} epoch = {epoch}:  mse = {self.avg_mse:0.4f}  |  rms = {self.avg_rms:0.4f}")

class CacheHistory:
    """ Cache history for all epochs.
    """
    def __init__(self, tag):
        """ Inintialize cache history (dict).

        Args:
            tag (str): "train" or "test"
        """
        self.tag = tag
        self.history = {
            "epoch": [],
            "mse": [],
            "rms": [],
        }

    def record_and_save(self, epoch, cache_epoch, CONFIG):
        """ Record all caches within a single epoch. Save history dict.

        Args:
            epoch (int): current epoch
            cache_epoch (CacheEpoch): cache_epoch object 
            CONFIG (dict): CONFIG
        """
        self.history["epoch"].append(epoch)
        self.history["mse"].append(cache_epoch.avg_mse)
        self.history["rms"].append(cache_epoch.avg_rms)

        fname = f"{CONFIG['output_folder']}/{self.tag}_history.npy"
        np.save(fname, self.history)