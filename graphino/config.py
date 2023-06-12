import numpy as np
import itertools
import qiskit
import os

#----------------------- Quantum Circuit Settings -----------------------
NUM_QUBITS      = 4
NUM_SHOTS       = 1 # for timing reasons is set to 1, but in IRL you want this value to be higher https://quantumcomputing.stackexchange.com/questions/9823/what-is-meant-with-shot-in-quantum-computation
NUM_LAYERS      = 2
SHIFT           = np.pi/4

def create_QC_OUTPUTS():
    measurements = list(itertools.product([0, 1], repeat=NUM_QUBITS))
    return [''.join([str(bit) for bit in measurement]) for measurement in measurements]

QC_OUTPUTS      = create_QC_OUTPUTS()
NUM_QC_OUTPUTS  = len(QC_OUTPUTS)

SIMULATOR       = qiskit.Aer.get_backend('qasm_simulator')