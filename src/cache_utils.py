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
        self._cum_loss = 0.0
        self._cum_err_mean = 0
        self._cum_err_std = 0

        self.avg_loss = 0.0
        self.avg_err_mean = 0
        self.avg_err_std = 0

    def update_cache(self, pred_mu, target, loss):
        """ Update cache for a learning update of a batch to the
            cummulative attributes.

        Args:
            pred_mu (torch.Tensor): predictions of the batch samples
            target (torch.Tensor): targets of the batch samples
            loss (torch.Tensor): weighted averaged loss of the batch
        """
        err = (pred_mu - target).cpu().detach().numpy()
        self._cum_err_mean += err.mean(axis=0)
        self._cum_err_std += err.std(axis=0)
        self._cum_loss += loss.item()
        self._cnt += 1

    def calc_avg_across_batches(self):
        """ Calculate the average of every attribute within a complete
            epoch (accross all batches).
        """
        self.avg_err_mean = self._cum_err_mean / self._cnt
        self.avg_err_std = self._cum_err_std / self._cnt
        self.avg_loss = self._cum_loss / self._cnt

    def print_cache(self, epoch):
        """ Print cache attributes (average over epoch).

        Args:
            epoch (int): epoch
        """
        avg_err_mean_print = np.array_str(self.avg_err_mean, precision=4)
        avg_err_std_print = np.array_str(self.avg_err_std, precision=4)
        print(f"\nepoch = {epoch}, {self.tag}:")
        print(f"    loss = {self.avg_loss:.4f}")
        print(f"        err_mean = {avg_err_mean_print}")
        print(f"        err_std = {avg_err_std_print}")


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
            "loss": [],
            "err_mean": [],
            "err_std": [],
        }

    def record_and_save(self, epoch, cache_epoch, CONFIG):
        """ Record all caches within a single epoch. Save history dict.

        Args:
            epoch (int): current epoch
            cache_epoch (CacheEpoch): cache_epoch object 
            CONFIG (dict): CONFIG
        """
        self.history["epoch"].append(epoch)
        self.history["loss"].append(cache_epoch.avg_loss)
        self.history["err_mean"].append(cache_epoch.avg_err_mean.tolist())
        self.history["err_std"].append(cache_epoch.avg_err_std.tolist())

        fname = f"{CONFIG['output_folder']}/{self.tag}_history.npy"
        np.save(fname, self.history)