#!/bin/bash
OP=$1

capitalize() {
    if [[ ${BASH_VERSINFO[0]} -ge 4 ]]; then
        printf '%s' "${1^}"
    else
        printf '%s%s' "$(printf '%s' "${1:0:1}" | tr '[:lower:]' '[:upper:]')" "${1:1}"
    fi
}

OP_DOC="$OP"-syntax.md
OP_GO="$OP".go

OP_CAP=$(capitalize "$OP")
TEST_NAME=Test"$OP_CAP"

echo 'See @'$OP_DOC' and @'$OP_GO'. Make sure we return directly the internal Go type `'$OP'` as `interface{}` type, and not Result wrapper objects. Remove any Result wrapper objects if implemented. Implement complete unit tests which check all of features mentioned in the documentation for @'$OP_GO' . Implement all tests, even for features which have not been implemented yet. Once unit tests are ready, they act as a specification. Run `go test -v -run "^'$TEST_NAME'" ./pkg/...` to run these tests. Then fix the implementation if tests are broken. Also use `gh` to check issue 140 for proper error handling. Fix the implementation to follow correct error handling.'
