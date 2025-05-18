#!/bin/bash

ls pkg/primitives/*.go|grep -v _| while read LINE; do if grep -q primitive_services.RegisterPrimitive $LINE; then :; else echo $LINE; fi; done
