KB = """
% Enter your KB below this line:

% R1:
problem(battery) :- battery(bad), not(engine(turn_over)).
% R2:
battery(bad) :- lights(weak).
% R3:
battery(bad) :- radio(weak).
% R4:
problem(engine_flooded) :- engine(turn_over), smell(gas).
% R5:
problem(out_of_gas) :- engine(turn_over), gas_gauge(empty).

engine(X) :- ask(engine, X).
smell(X) :- ask(smell, X).
gas_gauge(X) :- ask(gas_gauge, X).
lights(X) :- ask(lights, X).
radio(X) :- ask(radio, X).


% The code below implements the prompting to ask the user:


% Asking clauses

ask(A, V):-
known(yes, A, V), % succeed if true
!.	% stop looking

ask(A, V):-
known(_, A, V), % fail if false
!, fail.

ask(A, V):-
not multivalued(A),
write_py(A:not_multivalued),
known(yes, A, V2),
V \== V2,
!, fail.

ask(A, V):-
read_py(A,V,Y), % get the answer
asserta(known(Y, A, V)), % remember it
Y == yes.	% succeed or fail
"""

with open("KB_A.pl", "w") as text_file:
    text_file.write(KB)

# The code here will ask the user for input based on the askables
# It will check if the answer is known first
# import pip
# pip.main(['install', 'pyswip'])
# import subprocess
# subprocess.run(['sudo', 'pip', 'install', ''])

from pyswip.prolog import Prolog
from pyswip.easy import *

prolog = Prolog() # Global handle to interpreter

retractall = Functor("retractall")
known = Functor("known",3)

# Define foreign functions for getting user input and writing to the screen
def write_py(X):
    print(str(X))
    sys.stdout.flush()
    return True

def read_py(A,V,Y):
    Y.unify(raw_input(str(A) + " is " + str(V) + "?"))
    return True


write_py.arity = 1
read_py.arity = 3

registerForeign(read_py)
registerForeign(write_py)

prolog.consult("KB_A.pl") # open the KB
call(retractall(known))
for soln in prolog.query("problem(X).", maxresult=1):
    print("Your problem is " + soln['X'])


# NameError                                 Traceback (most recent call last)
# _ctypes/callbacks.c in 'calling callback function'()
# /home/user/pyswip/easy.pyc in wrapper(*args)
#     463         def wrapper(*args):
#     464             args = [getTerm(arg) for arg in args]
# --> 465             r = fun(*args)
#     466             return (r is None) and True or r
#     467         res = wrapper
# <ipython-input-7-0412197c9fb7> in read_py(A, V, Y)
#      21
#      22 def read_py(A,V,Y):
# ---> 23     Y.unify(raw_input(str(A) + " is " + str(V) + "?"))
#      24     return True
#      25
# /home/user/pyswip/easy.pyc in unify(self, value)
#     153     def unify(self, value):
#     154         if type(value) == str:
# --> 155             fun = PL_unify_atom_chars
#     156         elif type(value) == int:
#     157             fun = PL_unify_integer
# NameError: global name 'PL_unify_atom_chars' is not defined
