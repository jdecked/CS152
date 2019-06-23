
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
