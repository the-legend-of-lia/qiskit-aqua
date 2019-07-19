# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
The Deutsch-Jozsa algorithm.
"""

import logging
import operator
import numpy as np

from qiskit import ClassicalRegister, QuantumCircuit

from qiskit.aqua import AquaError, Pluggable, PluggableType, get_pluggable_class
from qiskit.aqua.algorithms import QuantumAlgorithm
from qiskit.aqua.utils import get_subsystem_density_matrix

from qiskit.assertions.assertmanager import AssertManager

logger = logging.getLogger(__name__)


class DeutschJozsa(QuantumAlgorithm):
    """The Deutsch-Jozsa algorithm."""

    CONFIGURATION = {
        'name': 'DeutschJozsa',
        'description': 'Deutsch Jozsa',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'dj_schema',
            'type': 'object',
            'properties': {
            },
            'additionalProperties': False
        },
        'problems': ['functionevaluation'],
        'depends': [
            {
                'pluggable_type': 'oracle',
                'default': {
                     'name': 'TruthTableOracle',
                },
            },
        ],
    }

    def __init__(self, oracle):
        self.validate(locals())
        super().__init__()

        self._oracle = oracle
        self._circuit = None
        self._breakpoints = []
        self._ret = {}

    @classmethod
    def init_params(cls, params, algo_input):
        if algo_input is not None:
            raise AquaError("Input instance not supported.")

        oracle_params = params.get(Pluggable.SECTION_KEY_ORACLE)
        oracle = get_pluggable_class(
            PluggableType.ORACLE,
            oracle_params['name']).init_params(params)
        return cls(oracle)

    def construct_circuit(self, measurement=False):
        """
        Construct the quantum circuit

        Args:
            measurement (bool): Boolean flag to indicate if measurement should be included in the circuit.

        Returns:
            the QuantumCircuit object for the constructed circuit
        """

        if self._circuit is not None:
            return self._circuit

        measurement_cr = ClassicalRegister(len(self._oracle.variable_register), name='m')

        # preoracle circuit
        qc_preoracle = QuantumCircuit(
            self._oracle.variable_register,
            self._oracle.output_register,
            measurement_cr
        )

        # classical input state in oracle variable_register
        self._breakpoints.append(qc_preoracle.assert_classical(self._oracle.variable_register, measurement_cr, .05, 0))

        qc_preoracle.h(self._oracle.variable_register)

        # uniform superposition in oracle variable_register after Hadamards
        self._breakpoints.append(qc_preoracle.assert_uniform(self._oracle.variable_register, measurement_cr, .05))

        qc_preoracle.x(self._oracle.output_register)
        qc_preoracle.h(self._oracle.output_register)
        qc_preoracle.barrier()

        # oracle circuit
        qc_oracle = self._oracle.circuit

        # postoracle circuit
        qc_postoracle = QuantumCircuit(
            self._oracle.variable_register,
            self._oracle.output_register,
            measurement_cr
        )
        qc_postoracle.h(self._oracle.variable_register)
        qc_postoracle.barrier()

        self._circuit = qc_preoracle + qc_oracle + qc_postoracle
        # classical output state in oracle variable_register after Hadamards
        self._breakpoints.append(self._circuit.assert_classical(self._oracle.variable_register, measurement_cr, .05, 0))
        self._breakpoints.append(self._circuit.assert_classical(self._oracle.variable_register, measurement_cr, .05, '100'))

        # measurement circuit
        if measurement:
            self._circuit.measure(self._oracle.variable_register, measurement_cr)

        # return self._circuit
        return self._breakpoints, self._circuit

    def _run(self):
        if self._quantum_instance.is_statevector:
            bp, qc = self.construct_circuit(measurement=False)
            result = self._quantum_instance.execute(qc)
            complete_state_vec = result.get_statevector(qc)
            variable_register_density_matrix = get_subsystem_density_matrix(
                complete_state_vec,
                range(len(self._oracle.variable_register), qc.width())
            )
            variable_register_density_matrix_diag = np.diag(variable_register_density_matrix)
            max_amplitude = max(
                variable_register_density_matrix_diag.min(),
                variable_register_density_matrix_diag.max(),
                key=abs
            )
            max_amplitude_idx = np.where(variable_register_density_matrix_diag == max_amplitude)[0][0]
            top_measurement = np.binary_repr(max_amplitude_idx, len(self._oracle.variable_register))
        else:
            bp, qc = self.construct_circuit(measurement=True)
            sim_result = self._quantum_instance.execute( bp + [qc] )

            # stat_outputs = AssertManager.stat_collect(qc[0:-1], sim_result)

            # assert classical input state in oracle variable_register
            print ( "sim_result.get_assert(bp[0]) = " )
            print ( sim_result.get_assertion_passed(bp[0]) )
            assert ( sim_result.get_assertion_passed(bp[0]) )

            # assert uniform superposition in oracle variable_register after Hadamards
            assert ( sim_result.get_assertion_passed(bp[1]) )

            # assert classical output state in oracle variable_register after Hadamards
            assert ( sim_result.get_assertion_passed(bp[2]) != sim_result.get_assertion_passed(bp[3]) )

            measurement = sim_result.get_counts(qc)
            print ("measurement = ")
            print (measurement)
            self._ret['measurement'] = measurement
            top_measurement = max(measurement.items(), key=operator.itemgetter(1))[0]

        self._ret['result'] = 'constant' if int(top_measurement) == 0 else 'balanced'

        return self._ret
