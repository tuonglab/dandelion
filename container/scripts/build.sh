# Local build/test/push of the sc-dandelion image.
# Run from the repository root. CI does the same in
# .github/workflows/container-build.yml.

# build (release variant)
docker build --platform linux/amd64 -f container/Dockerfile \
    --build-arg DANDELION_VERSION="$(python -m setuptools_scm)" \
    -t sc-dandelion:local .

# devel/testing variant: tolerate database download failures, also build OGRDB
# docker build --platform linux/amd64 -f container/Dockerfile \
#     --build-arg ALLOW_DB_FAILURE=1 --build-arg INCLUDE_OGRDB=1 \
#     -t sc-dandelion:dev .

# test
docker run --rm sc-dandelion:local pytest -p no:cacheprovider /tests \
    -W ignore::DeprecationWarning -W ignore::PendingDeprecationWarning -W ignore::FutureWarning

# test the HPC route
apptainer build sc-dandelion.sif docker-daemon://sc-dandelion:local
apptainer run -B $PWD sc-dandelion.sif dandelion-preprocess --help

# push (normally done by the "Release Container Image" workflow)
# docker login docker.io
# docker tag sc-dandelion:local zktuong/sc-dandelion:<tag>
# docker push zktuong/sc-dandelion:<tag>
