#!/bin/bash

s=$1

if [ "$s" == "beta" ] ; then
   echo "beta detected"
fi

if [ "$s" == "lognormal" ] ; then
   echo "lognormal detected"
fi      
