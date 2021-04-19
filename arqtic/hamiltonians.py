import arqtic.program

class Hamiltonian:
    def __init__(self, nspins, terms):
        self.nspins = nspins
        self.terms = terms

    def add_term(self, term):
        self.term.append(term)

    def matrix(self):
        dim = 2**self.nspins
        ham_mat = np.zeros((dim,dim))
        for term in self.terms:
            kron_list = ['I'] * self.nspins
            for pauli in term.paulis:
               kron_list[pauli.qubit] = pauli.name
            for q in range(self.nspins-1):
                if (q==0):
                    mat = np.kron(gate_matrix_dict[kron_list[0]], gate_matrix_dict[kron_list[1]])
                else:          
                    mat = np.kron(mat, gate_matrix_dict[kron_list[q+1]])
            mat = term.coeff * mat 
            ham_mat = ham_mat + mat
        return ham_mat
    
    def show(self):
        dim = 2**self.nspins
        ham_mat = self.matrix()
        for i in range(dim):
            for j in range(dim):
                print(ham_mat[i][j])

class Ising_Hamiltonian:
    def __init__(self, nspins, exchange_coeff, ext_mag_vec, pbc=False):
        self.nspins = nspins
        self.exchange_coeff = exchange_coeff
        self.ext_mag_vec = ext_mag_vec
        self.pbc = pbc

    def matrix(self):
        ham_terms = []
        x_field = self.ext_mag_vec[0]
        y_field = self.ext_mag_vec[1]
        z_field = self.ext_mag_vec[2]
        #add Ising exchange interaction terms to Hamiltonian
        for q in range(self.nspins-1):
            pauli1 = Pauli('Z',q)
            pauli2 = Pauli('Z',q+1)
            paulis = [pauli1, pauli2]
            exch_term = Term(paulis, -1.0*self.exchange_coeff)
            ham_terms.append(exch_term)
        #in case of pbc=true add term n->0
        if(self.pbc): 
            pauli1 = Pauli('Z',self.nspins-1)
            pauli2 = Pauli('Z',0)
            paulis = [pauli1, pauli2]
            exch_term = Term(paulis, -1.0*self.exchange_coeff)
            ham_terms.append(exch_term)
        #add external magnetic field terms to Hamiltonian
        for q in range(self.nspins):
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
            for q in range(0, self.nspins):
                instr_set1.append(Gate([q], 'H'))
                instr_set1.append(Gate([q], 'RZ', angles=[(-2.0*self.ext_mag_vec[0]*delta_t/H_BAR)]))
                instr_set1.append(Gate([q],'H'))
            for q in range(0, self.nspins-1):
                instr_set2.append(Gate([q, q+1], 'CNOT'))
                instr_set2.append(Gate([q+1],'RZ', angles=[-2.0*self.exchange_coeff*delta_t/H_BAR]))
                instr_set2.append(Gate([q, q+1], 'CNOT'))
            p.add_instr(instr_set1)
            p.add_instr(instr_set2)
        return p
