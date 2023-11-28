#!/usr/bin/bash
echo -n "username:"
read USERNAME

BASEIMAGE_NAME=lumi-pytorch-rocm-5.5.1-python-3.10-pytorch-v2.0.1.sif

if [ ! -f $BASEIMAGE_NAME ]
then
scp -C $USERNAME@lumi.csc.fi:/appl/local/containers/sif-images/$BASEIMAGE_NAME .
fi

singularity build --fakeroot klajster.sif klajster.def
scp -C klajster.sif $USERNAME@lumi.csc.fi:/project/project_465000858/

