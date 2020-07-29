from .program import Program, Pauli, Term, Gate

class Hamiltonian:
    def __init__(self, nqubits, terms):
        self.nqubits = nqubits
        self.terms = terms

    def add_term(self, term):
        self.term.append(term)

    def matrix(self):
        dim = 2**self.nqubits
        ham_mat = np.zeros((dim,dim))
        for term in self.terms:
            kron_list = ['I'] * self.nqubits
            for pauli in term.paulis:
               kron_list[pauli.qubit] = pauli.name
            for q in range(self.nqubits-1):
                if (q==0):
                    mat = np.kron(gate_matrix_dict[kron_list[0]], gate_matrix_dict[kron_list[1]])
                else:          
                    mat = np.kron(mat, gate_matrix_dict[kron_list[q+1]])
            mat = term.coeff * mat 
            ham_mat = ham_mat + mat
        return ham_mat
    
    def show(self):
        dim = 2**self.nqubits
        ham_mat = self.matrix()
        for i in range(dim):
            for j in range(dim):
                print(ham_mat[i][j])

class Ising_Hamiltonian:
    def __init__(self, nqubits, exchange_coeff, ext_mag_vec, pbc=False):
        self.nqubits = nqubits
        self.exchange_coeff = exchange_coeff
        self.ext_mag_vec = ext_mag_vec
        self.pbc = pbc

    def matrix(self):
        ham_terms = []
        x_field = self.ext_mag_vec[0]
        y_field = self.ext_mag_vec[1]
        z_field = self.ext_mag_vec[2]
        #add Ising exchange interaction terms to Hamiltonian
        for q in range(self.nqubits-1):
            pauli1 = Pauli('Z',q)
            pauli2 = Pauli('Z',q+1)
            paulis = [pauli1, pauli2]
            exch_term = Term(paulis, -1.0*self.exchange_coeff)
            ham_terms.append(exch_term)
        #in case of pbc=true add term n->0
        if(self.pbc): 
            pauli1 = Pauli('Z',self.nqubits-1)
            pauli2 = Pauli('Z',0)
            paulis = [pauli1, pauli2]
            exch_term = Term(paulis, -1.0*self.exchange_coeff)
            ham_terms.append(exch_term)
        #add external magnetic field terms to Hamiltonian
        for q in range(self.nqubits):
            if (x_field != 0.0):
                pauli = Pauli('X', q)
                term = Term([pauli], -1.0*x_field)
                ham_terms.append(term)
            if (y_field != 0.0):
                pauli = Pauli('Y', q)
                term = Term([pauli], -1.0*y_field)
                ham_terms.append(term)
            if (z_field != 0.0):
                pauli = Pauli('Z', q)
                term = Term([pauli], -1.0*z_field)
                ham_terms.append(term)
        ham_mat = Hamiltonian(self.nqubits, ham_terms)
        return ham_mat.matrix()

    def show(self):
        dim = 2**self.nqubits
        ham_mat = self.matrix()
        for i in range(dim):
            for j in range(dim):
                print(ham_mat[i][j])

    def print_pretty(self):
        dim = 2**self.nqubits
        ham_mat = self.matrix()
        for i in range(dim):
            print(list(ham_mat[i]), ",")

    def get_Trotter_program(self, delta_t, total_time): #right now only works for x-dir external magnetic field and pbc=False
        H_BAR = 0.658212 #eV*fs 
        p = Program(self.nqubits)
        timesteps = int(total_time/delta_t)
        for t in range(0,timesteps):
            instr_set1 = []
            instr_set2 = []
            for q in range(0, self.nqubits):
                instr_set1.append(Gate('H', [q]))
                instr_set1.append(Gate('RZ', [q], angles=[(-2.0*self.ext_mag_vec[0]*delta_t/H_BAR)]))
                instr_set1.append(Gate('H',[q]))
            for q in range(0, self.nqubits-1):
                instr_set2.append(Gate('CNOT',[q, q+1]))
                instr_set2.append(Gate('RZ', [q+1], angles=[-2.0*self.exchange_coeff*delta_t/H_BAR]))
                instr_set2.append(Gate('CNOT', [q, q+1]))
            p.add_instr(instr_set1)
            p.add_instr(instr_set2)
        return p
