#!/bin/bash

version=$(cat version.txt)

if [[ $version =~ ^[0-9]+\.[0-9]+\.[0-9]+\.dev[0-9]+$ ]]; then
  echo "Valid dev version."
  exit 0
else
  echo "Invalid dev version! Add .dev[xx] suffix in version.txt."
  exit 1
fi