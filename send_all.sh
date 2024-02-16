#!/bin/bash

git add .

read -p "comment for pushing: " comment

git commit -m "$comment"

git push

