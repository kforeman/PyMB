#!/bin/bash
set -e # Exit with nonzero exit code if anything fails

R -e 'library(TMB); runExample("randomregression");