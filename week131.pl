% % s  -->  foo,bar,wiggle.
% % foo  -->  [choo].
% % foo  -->  foo,foo.
% % bar  -->  mar,zar.
% % mar  -->  me,my.
% % me  -->  [i].
% % my  -->  [am].
% % zar  -->  blar,car.
% % blar  -->  [a].
% % car  -->  [train].
% % wiggle  -->  [toot].
% % wiggle  -->  wiggle,wiggle.
% % "choo i am a train toot | toot toot | toot toot toot"
%
% % translation
% % s(A, B) :- foo(A, C), bar(C, D), wiggle(D, B).
% % foo(A, B) :- 'C'(A, choo, B).
% % foo(A, B) :- foo(A, C), foo(C, B).
% % bar(A, B) :- mar(A, C), zar(C, B).
% % mar(A, B) :- me(A, C), my(C, B).
% % me(A, B) :- 'C'(A, i, B).
% % my(A, B) :- 'C'(A, am, B).
% % zar(A, B) :- blar(A, C), car(C, B).
% % blar(A, B) :- 'C'(A, a, B).
% % car(A, B) :- 'C'(A, train, B).
% % wiggle(A, B) :- 'C'(A, toot, B).
% % wiggle(A, B) :- wiggle(A, C), wiggle(C, B).
%
% % s(A,B)  :-  [].
% % s(A,B) :- [ab], s(A,B).
%
% % l  -->  [a].
% % r  -->  [b].
%
% s --> np, vp.
% np --> pn.
% np --> det, n.
% np --> det, adj, n.
% vp --> iv.
% vp --> tv, np.
%
%
% det --> [the].
%
% n --> [dog, cat, snake, dog].
% % n --> [cat].
% % n --> [snake].
% % n --> [dog].
%
% pn --> [cedric].
% pn --> [cecilia].
%
% adj --> [fast].
% adj --> [lazy].
%
% tv --> [chases].
% tv --> [befriends].
%
% iv --> [sleeps].

run_command(TOKENS, COMMAND) :-
   command(CLIST, TOKENS, []),
   COMMAND =.. CLIST,
   call(COMMAND).

% Define the exit predicate to do nothing.
exit.

%%% DCGs

command([OP|ARGS]) --> operation(OP), arguments(ARGS).

arguments([ARG|ARGS]) --> argument(ARG), arguments(ARGS).
arguments([]) --> [].

operation(report) --> [list].
operation(book) --> [book].
operation(cancel) --> [cancel].
operation(exit) --> ([exit]; [quit]; [bye]).

argument(passengers) --> [passengers].
argument(flights) --> [flights].

argument(FLIGHT) --> [FLIGHT], {flight(FLIGHT)}.
argument(PASSENGER) --> [PASSENGER].

% Flights

flight(aa101).
flight(sq238).
flight(mi436).
flight(oz521).

% Command predicates

report(flights) :-
   flight(F),
   write(F),
   fail.
report(_).

report(passengers, FLIGHT) :-
   booked(PASSENGER, FLIGHT),
   write(PASSENGER),
   fail.
report(_, _).

book(PASSENGER, FLIGHT) :-
   assertz(booked(PASSENGER, FLIGHT)).

cancel(PASSENGER, FLIGHT) :-
   retract(booked(PASSENGER, FLIGHT)).
