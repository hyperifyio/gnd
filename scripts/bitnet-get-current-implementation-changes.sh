#!/bin/bash
git diff bitnet $(git diff bitnet --name-only pkg/bitnet|grep -vF _test|grep -vF /testdata/|cat)|cat
