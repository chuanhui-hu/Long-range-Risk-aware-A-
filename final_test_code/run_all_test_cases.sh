#!/bin/bash

for n in {0..9}
do
  python3 path_planning_test.py $n 4 2
  python3 path_planning_test.py $n 4 200
done

echo All done

