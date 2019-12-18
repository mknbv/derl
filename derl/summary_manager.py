""" Defines summary manager. """
from torch.utils.tensorboard import SummaryWriter


_DEFAULT_SUMMARY_MANAGER = None


class SummaryManager:
  """ Extends summary writer with step and log period. """
  def __init__(self, logdir, log_period, step_var, **kwargs):
    self.summary_writer = SummaryWriter(logdir, **kwargs)
    self.log_period = log_period
    self.step_var = step_var
    self.last_recorded_step = self.step_var

  def should_record_summaries(self):
    """ Returns true if summaries should be recorded. """
    if (self.step_var == self.last_recorded_step
        or self.step_var - self.last_recorded_step >= self.log_period):
      self.last_recorded_step = self.step_var
      return True
    return False

  def set_as_default(self):
    """ Sets summary manager as default. """
    set_summary_manager(self)


def get_summary_manager():
  """ Returns summary manager. """
  return _DEFAULT_SUMMARY_MANAGER

def set_summary_manager(summary_manager):
  """ Sets summary manager. """
  global _DEFAULT_SUMMARY_MANAGER # pylint: disable=global-statement
  _DEFAULT_SUMMARY_MANAGER = summary_manager

def should_record_summaries():
  """ Returns True if summary manager should record summaries. """
  summary_manager = get_summary_manager()
  return (summary_manager is not None
          and summary_manager.should_record_summaries())


def foreward_to_summary_manager(func):
  """ Calls function with the same name of the summary writer instance. """
  def wrapped(*args, **kwargs):
    summary_manager = get_summary_manager()
    if summary_manager is None:
      raise ValueError(f"trying to call function {func} when summary manager "
                       "is not set")
    summary_writer_func = getattr(summary_manager.summary_writer, func.__name__)
    summary_writer_func(*args, **kwargs)
  return wrapped


# pylint: disable=unused-argument
@foreward_to_summary_manager
def add_scalar(tag, scalar_value, global_step=None, walltime=None):
  """ Adds scalar summary. """
