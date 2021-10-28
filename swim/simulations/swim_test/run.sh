#!/bin/bash
MAINSIMDIR=../../src/
MAINSIMEXEC=swim

if [ $# -lt 1 ]; then
	echo "usage: $0 config [run-number(s)|all [ini-file]]"
	echo example:
	echo "  "$0 Reactive 0..2
	exit 1
fi

RUNS=""
if [ "$2" != "" ] && [ "$2" != "all" ]; then
    RUNS="-r $2"
fi

INIFILE="swim_test.ini"
if [ "$3" != "" ]; then
    INIFILE="$3"
fi

python NewMPCController.py >MPC.log &
PID=$!
sleep 2s
opp_runall -j1 $MAINSIMDIR/$MAINSIMEXEC $INIFILE -u Cmdenv -c $1 -n ..:$MAINSIMDIR:../../../queueinglib:../../src -lqueueinglib $RUNS

sleep 3s
kill -9 $PID