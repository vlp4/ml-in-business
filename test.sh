#!/bin/bash

curl -v -D - -S --request POST --header "Content-Type: application/json" \
  --data '{"age":35,"workclass":"Private","fnlwgt":1,"education":"Bachelors","education-num":1,"marital-status":"Married-civ-spouse","occupation":"Sales","relationship":"Husband","race":"Black","sex":"Male","capital-gain":1,"capital-loss":0,"hours-per-week":40,"native-country":"Cambodia"}' \
  http://localhost:8180/predict