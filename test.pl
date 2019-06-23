% We need this module to use some of the constraint programming extensions to Prolog
:- use_module(library(clpfd)).


% Pss is a list of 9 rows representing the game board.
% flatten(X, Y) flattens X, a nested list, into Y, an un-nested list.
% "x ins 1..9" constraints all elements x to the domain 1..9
% maplist(X, Y) maps a goal X onto every element in the list Y — but only if this goal X can be applied to every item in L.
% columns(X) - builds vertical columns from the lists given (zip(X,Y,Z)). Thats why we use the list [H|....] every time etracting one column at a time
% label(X) instead of printing prologs internal variables, prints whatever theyve unified with
% extend to 4x4 by


sudoku(Pss) :-
    flatten(Pss, Ps),
    Ps ins 1..9,
    maplist(all_distinct, Pss),
    Pss = [R1,R2,R3,R4,R5,R6,R7,R8,R9],
    columns(R1, R2, R3, R4, R5, R6, R7, R8, R9),
    blocks(R1, R2, R3), blocks(R4, R5, R6), blocks(R7, R8, R9),
    label(Ps).

columns([], [], [], [], [], [], [], [], []).
columns([A|As],[B|Bs],[C|Cs],[D|Ds],[E|Es],[F|Fs],[G|Gs],[H|Hs],[I|Is]) :-
    all_distinct([A,B,C,D,E,F,G,H,I]),
    columns(As, Bs, Cs, Ds, Es, Fs, Gs, Hs, Is).

blocks([], [], []).
blocks([X1,X2,X3|R1], [X4,X5,X6|R2], [X7,X8,X9|R3]) :-
    all_distinct([X1,X2,X3,X4,X5,X6,X7,X8,X9]),
    blocks(R1, R2, R3).

problem(1, [[_,_,5, _,_,7, _,_,2],
            [_,7,_, _,8,_, _,4,_],
            [8,_,_, 1,_,_, 3,_,_],

            [6,_,_, 9,_,_, 5,_,_],
            [_,9,_, _,3,_, _,8,_],
            [_,_,3, _,_,8, _,_,6],

            [_,_,4, _,_,1, _,_,8],
            [_,3,_, _,5,_, _,7,_],
            [1,_,_, 3,_,_, 6,_,_]]).


:- problem(1, Rows), sudoku(Rows), maplist(writeln,Rows), halt.
