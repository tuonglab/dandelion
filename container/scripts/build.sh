# release
sudo singularity build --notest sc-dandelion.sif sc-dandelion.def
sudo singularity test --writable-tmpfs sc-dandelion.sif
singularity sign sc-dandelion.sif
singularity verify --local sc-dandelion.sif
# Sylabs Cloud is decommissioned; push to Docker Hub as an ORAS artifact instead.
# Log in once with: singularity registry login --username <user> oras://docker.io
singularity push sc-dandelion.sif oras://docker.io/zktuong/sc-dandelion:latest

# devel and testing
sudo singularity build --notest sc-dandelion_dev.sif sc-dandelion_dev.def
sudo singularity test --writable-tmpfs sc-dandelion_dev.sif
singularity sign sc-dandelion_dev.sif
singularity verify --local sc-dandelion_dev.sif
