% hasWand(harry).
% quidditchPlayer(harry).
% wizard(ron).
% wizard(X):- hasBroom(X), hasWand(X).
% hasBroom(X):- quidditchPlayer(X).

% Predicates
enrolled(jack, cs131).
enrolled(charlie, cs135).
enrolled(olivia, cs135).
enrolled(arthur, cs134).
enrolled(jason, cs171).
enrolled(monty, cs171).
teaches(collins, cs131).
teaches(collins, cs171).
teaches(kirke, cs135).
teaches(juniper, cs134).

% Rules
professor(X, Y) :- enrolled(Y, Z), teaches(X, Z).

students() :-
    findall(X, enrolled(X, cs135), L),
    format('~w and ~w are students in cs135.', L).

professors() :-
    findall([X, Y], professor(X, Y), L),
    forall(
        member([X, Y], L),
        format('prof. ~w teaches ~w.\n', [X, Y])
    ).
