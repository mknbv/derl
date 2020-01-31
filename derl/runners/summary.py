""" Implements runner wrapper that enables/disables summaries. """
from derl.runners.env_runner import RunnerWrapper
import derl.summary as summary


class PeriodicSummaries(RunnerWrapper):
  """ Enables summary recording with specified period. """
  def __init__(self, runner, log_period):
    super().__init__(runner)
    self.log_period = log_period
    self.last_record_step = None

  @classmethod
  def make_with_nlogs(cls, runner, nlogs=1e5):
    """ Creates an instance with specified number of logs to be written. """
    if runner.nsteps is None:
      raise ValueError("runner.nsteps cannot be None")
    log_period = int(runner.nsteps / nlogs)
    return cls(runner, log_period)

  def run(self, obs=None):
    summary.start_recording()
    self.last_record_step = int(self.runner.step_var)
    for interactions in self.runner.run(obs):
      yield interactions
      next_step = int(self.runner.step_var) + 1
      should_record = next_step - self.last_record_step >= self.log_period
      summary.set_recording(should_record)
      if should_record:
        self.last_record_step = next_step
