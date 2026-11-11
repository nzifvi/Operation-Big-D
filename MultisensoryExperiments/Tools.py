import numpy


class EEG:
    """
    Logs the membrane voltage V of selected neurons at every simulation step.

    details is a dict of the form:
        {
            "popName" : (pop, monitoredNeurons)
        }
    where pop is a pygenn NeuronGroup with a variable "V" and monitoredNeurons
    is a list of neuron indices within that population.

    Call log(model.t) once after every model.step_time().
    """
    def __init__(self, details) -> None:
        self._details = {
            popName: (pop, numpy.asarray(monitoredNeurons, dtype=int))
            for popName, (pop, monitoredNeurons) in details.items()
        }
        self._times = []
        self._log   = self._initLogger()

    def _initLogger(self) -> dict:
        # One list per population; each step appends a row of voltages,
        # one entry per monitored neuron.
        return {popName: [] for popName in self._details}

    def log(self, time) -> None:
        self._times.append(time)
        for popName, (pop, monitoredNeurons) in self._details.items():
            V = pop.vars["V"]
            V.pull_from_device()
            self._log[popName].append(V.view[monitoredNeurons].copy())

    def getLog(self) -> tuple:
        """
        Returns (times, voltages) where times has shape (steps,) and
        voltages[popName] has shape (steps, len(monitoredNeurons)).
        """
        times = numpy.asarray(self._times)
        voltages = {
            popName: numpy.vstack(rows) if rows
                     else numpy.empty((0, len(self._details[popName][1])))
            for popName, rows in self._log.items()
        }
        return times, voltages

    def getTrace(self, popName, neuron) -> tuple:
        """Returns (times, V) for one monitored neuron, ready for plt.plot."""
        times, voltages = self.getLog()
        column = int(numpy.flatnonzero(self._details[popName][1] == neuron)[0])
        return times, voltages[popName][:, column]