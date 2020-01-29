""" Defines summary object which stores summaries. """
from inspect import getmembers, isfunction
import sys
from torch.utils.tensorboard import SummaryWriter



class PeriodicRecorder:
  """ Signalizes wether recording should be done with a period. """
  def __init__(self, period, step_var):
    self.period = period
    self.step_var = step_var
    self.last_record_step = int(self.step_var) - self.period

  def __call__(self):
    step = int(self.step_var)
    should_record = (step == self.last_record_step
                     or step - self.last_record_step >= self.period)
    if should_record:
      self.last_record_step = step
    return should_record


def const(value):
  """ Function with constant value. """
  return lambda: value


class Summary:
  """ Caches summaries to be passed to summary_writer. """
  def __init__(self, module):
    self.module = module
    self.writer = None
    self.should_record_fn = const(True)
    self.set_writer_functions()

  def should_record(self):
    """ Returns true if summaries should be recorded during this step. """
    return self.should_record_fn()

  def start_recording(self):
    """ Enables summary recording. """
    self.should_record_fn = const(True)

  def stop_recording(self):
    """ Disables summary recording. """
    self.should_record_fn = const(False)

  def set_recording(self, flag):
    """ Sets recording on or off depending on boolean flag. """
    self.should_record_fn = const(flag)

  def record_with_period(self, period, step_var):
    """ Automatically start and stop recording with period. """
    self.should_record = PeriodicRecorder(period, step_var)

  def set_writer(self, summary_writer):
    """ Sets summary writer. """
    self.writer = summary_writer

  def make_writer(self, *args, **kwargs):
    """ Creates and sets summary writer. """
    self.set_writer(SummaryWriter(*args, **kwargs))

  def set_writer_functions(self):
    """ Adds summary writer functions to the class. """
    def wrap_func(func):
      def wrapped(*args, **kwargs):
        if self.writer is None:
          raise ValueError("summary.writer cannot be None, call set_writer or "
                           "make_writer to set writer")
        return getattr(self.writer, func.__name__)(*args, **kwargs)
      return wrapped

    for func in (value for name, value in getmembers(SummaryWriter)
                 if isfunction(value)):
      setattr(self, func.__name__, wrap_func(func))
    del self.__class__.set_writer_functions

  def __getattr__(self, name):
    return getattr(self.module, name)


sys.modules[__name__] = Summary(sys.modules[__name__])


# This is to prevent pylint from complaing that the module does not have
# functions added from `SummaryWriter`. Since overriding module attribute
# by providing module-level __getattr__ was only added in version 3.7 we
# keep the approach with wrapping the module above in order to support
# python 3.6 as well.
def __getattr__(name):
  return getattr(sys.modules[__name__], name)
