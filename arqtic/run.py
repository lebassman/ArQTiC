from MISTIQC import *
tester=Heisenberg('Domain_wall_example.txt')
tester.connect_account()
output=tester.return_circuits()

print("###########################")
print("Printing post-transpiled quantum circuit for debugging purposes")
print(output[2].qasm())
print("###########################")

# tester.run_circuits()

