import numpy as np


def apply_shifted_cnot(state, k):
	new_state = list(state)
	tgt = (5 - k) % 8
	ctrl = (2 - k) % 8
	new_state[tgt] ^= new_state[ctrl]
	return tuple(new_state)


def state_to_int(s):
	return sum(v * (2 ** (7 - i)) for i, v in enumerate(s))


def is_valid_sm_state(s):
	G0, G1, LQ, C0, C1, I3, chi, W = s
	if G0 == 1 and G1 == 1: return False
	if chi != W: return False
	if LQ == 0 and (C0 != 0 or C1 != 0): return False
	if LQ == 1 and (C0 == 0 and C1 == 0): return False
	if LQ == 0 and I3 == 0 and chi == 1: return False
	return True


def main():
	all_states = [tuple((i >> (7 - j)) & 1 for j in range(8)) for i in range(256)]
	delta = 2.0 / 9.0

	# 1. Base Quantum Walk Operator U
	A = np.zeros(8, dtype=complex)
	A[0] = np.sqrt(1.0 - delta)
	for k in range(1, 8):
		A[k] = np.sqrt(delta / 7.0) * np.exp(1j * k * np.pi / 4.0)

	U = np.zeros((256, 256), dtype=complex)
	for s_idx in range(256):
		s_tuple = all_states[s_idx]
		for k in range(8):
			t_tuple = apply_shifted_cnot(s_tuple, k)
			U[state_to_int(t_tuple), s_idx] += A[k]

	# 2. 4-Step Loop-Level Propagator
	M_tree = U.conj().T @ U
	M_loop = M_tree @ M_tree

	# 3. Construct the SU(3) Colour-Singlet Bases (Euclidean Action Limit)
	def get_color_singlet(G0, G1, I3):
		vec = np.zeros(256, dtype=complex)
		valid_colors = [(0, 1), (1, 0), (1, 1)]
		weights = []
		indices = []
		for C0, C1 in valid_colors:
			s = (G0, G1, 1, C0, C1, I3, 0, 0)
			if is_valid_sm_state(s):
				w = np.exp(-delta * (C0 + C1))
				weights.append(w)
				indices.append(state_to_int(s))
		norm = np.sqrt(sum(w ** 2 for w in weights))
		for i, idx in enumerate(indices):
			vec[idx] = weights[i] / norm
		return vec

	# Topologically ordered to perfectly match Eqs (2) and (3)
	gens = [(0, 0), (1, 0), (0, 1)]
	up_basis = [get_color_singlet(g[0], g[1], 0) for g in gens]
	dn_basis = [get_color_singlet(g[0], g[1], 1) for g in gens]

	# 4. Project Full Propagator onto Topological Bounds
	H_Up = np.zeros((3, 3), dtype=complex)
	H_Down = np.zeros((3, 3), dtype=complex)
	for i in range(3):
		for j in range(3):
			H_Up[i, j] = up_basis[i].conj().T @ M_loop @ up_basis[j]
			H_Down[i, j] = dn_basis[i].conj().T @ M_loop @ dn_basis[j]

	# 5. Exact Diagonalisation
	evals_U, U_u = np.linalg.eigh(H_Up)
	evals_D, U_d = np.linalg.eigh(H_Down)

	# The argmax assignment handles the eigenstate-to-generation mapping
	idx_U = np.argmax(np.abs(U_u), axis=1)
	evecs_U = U_u[:, idx_U]

	idx_D = np.argmax(np.abs(U_d), axis=1)
	evecs_D = U_d[:, idx_D]

	# 6. Extract the Unrenormalised physical CKM parameters
	V_CKM = evecs_U.conj().T @ evecs_D

	# Align the topological sequence to the physical SM mass ordering
	V_CKM_phys = V_CKM[[1, 0, 2]][:, [1, 0, 2]]

	print("PHYSICAL CKM MATRIX MAGNITUDES (|V_ij|):")
	for row in np.abs(V_CKM_phys):
		print("  [" + ", ".join([f"{x:8.3f}" for x in row]) + "]")

	J = np.imag(V_CKM_phys[0, 1] * V_CKM_phys[1, 2] * np.conj(V_CKM_phys[0, 2]) * np.conj(V_CKM_phys[1, 1]))
	print(f"\nJarlskog Invariant |J|: {abs(J):.3e}")


if __name__ == "__main__":
	main()
