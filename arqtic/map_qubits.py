def map_qubits(program, qubit_mapping):
    result = []
    for instr in program:
        if isinstance(instr, Gate):
            remapped_qubits = [qubit_mapping[q] for q in instr.qubits]
            gate = Gate(instr.name, instr.params, remapped_qubits)
            gate.modifiers = instr.modifiers
            result.append(gate)
        elif isinstance(instr, Measurement):
            result.append(Measurement(qubit_mapping[instr.qubit], instr.classical_reg)) 
        else:
            result.append(instr)
    new_program = program.copy()
    new_program._instructions = result

    return new_program
