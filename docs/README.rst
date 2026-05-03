|Docs| |PyPI| |Master| |MasterTest| |CodeCov| |Colab|

|logo|

Hi there! I have put together a python package for analyzing single cell
BCR/TCR data from multiple platforms, including 10x Genomics 5' VDJ,
BD Rhapsody, Parse Biosciences, SeekGene, and other AIRR-compatible
sequencing sources. It streamlines the
pre-processing, leveraging some tools from the immcantation suite, and
integrates with scanpy/anndata for single-cell BCR/TCR analysis. It also
includes a couple of functions for visualization. Try it out on |Colab| !

Also check out our review at `Nature Methods <https://www.nature.com/articles/s41592-024-02243-4>`__ on "Single-cell immune
repertoire analysis":

.. [Irac2024] Irac *et al.* (2021),
   *Single-cell immune repertoire analysis*,
   `Nature Methods <https://www.nature.com/articles/s41592-024-02243-4>`__.

Overview
--------

|overview|

Illustration of the ``Dandelion`` class slots

|class|

Please refer to the
`documentation <https://sc-dandelion.readthedocs.io/>`__ or the
notebooks
`here <https://nbviewer.jupyter.org/github/tuonglab/dandelion/tree/latest/docs/notebooks/>`__:

The raw files used in the example notebooks can be downloaded from 10X's
Single Cell Immune Profiling datasets
`website <https://support.10xgenomics.com/single-cell-vdj/datasets>`__.


.. raw:: html

    <div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden; max-width: 100%; height: auto;">
        <iframe src="https://www.youtube.com/embed/pxz31b1iIFY" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe>
    </div>


Installation
------------

Singularity container
~~~~~~~~~~~~~~~~~~~~~

``dandelion`` now comes ready in the form of a singularity container
which has all the required dependencies installed:

.. code:: bash

    singularity pull library://kt16/default/sc-dandelion:latest
    singularity shell --writable-tmpfs -B $PWD sc-dandelion_latest.sif


This will load up a conda-environment that has all the required
dependencies installed.

This can be used for the preprocessing steps by navigating to the data
folder and use:

.. code:: bash

    singularity run -B $PWD sc-dandelion_latest.sif dandelion-preprocess

Python package
~~~~~~~~~~~~~~

Start off by creating a conda environment containing scanpy, following
`official scanpy instructions <https://scanpy.readthedocs.io/en/stable/installation.html>`__.
Once done, run the following:

**Base install:**

.. code:: bash

    pip install sc-dandelion

**With Polars backend (recommended for improved performance):**

.. code:: bash

    pip install sc-dandelion[polars]

The ``polars`` extra enables a faster backend for data operations. The base install
uses ``pandas`` as the backend.

Between this and the pipelines within the singularity container, you
should be covered for most of your needs.

Manual full installation
~~~~~~~~~~~~~~~~~~~~~~~~

For those who cannot use the singularity container, here is a detailed
installation guide. Instructions may vary depending on your system.

First, install igblast and blast:

.. code:: bash

    conda install -c bioconda igblast blast
    # if the above doesn't work, download them manually:
    # https://ftp.ncbi.nih.gov/blast/executables/igblast/release/LATEST/
    # https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/
    echo 'export PATH=path/to/igblast/bin:$PATH' >> ~/.bash_profile
    echo 'export PATH=path/to/blast+/bin:$PATH' >> ~/.bash_profile

Download the germline databases using the
`scripts <https://github.com/tuonglab/dandelion/tree/master/container/scripts>`__
in the ``container`` folder:

.. code:: bash

    python prepare_imgt_database.py
    python prepare_ogrdb_database.py

Set the paths to the germline and igblast databases:

.. code:: bash

    echo 'export GERMLINE=path/to/database/germlines/' >> ~/.bash_profile
    echo 'export IGDATA=path/to/database/igblast/' >> ~/.bash_profile
    echo 'export BLASTDB=path/to/database/blast/' >> ~/.bash_profile
    source ~/.bash_profile

R packages for preprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For some preprocessing steps you will need ``rpy2`` and the following R
packages. The easiest way is via conda:

.. code:: bash

    conda install -c conda-forge -c bioconda rpy2 r-optparse r-alakazam r-tigger r-airr r-shazam r-scoper

Alternatively, install ``rpy2`` via pip and the R packages from within R:

.. code:: bash

    pip install rpy2
    # If compilation fails with clang, point to gcc first:
    # env CC=/path/to/bin/gcc-9 pip install rpy2

.. code:: r

    install.packages(c("optparse", "alakazam", "tigger", "airr", "shazam", "scoper"))

Finally, install dandelion:

.. code:: bash

    pip install sc-dandelion
    # or from GitHub
    pip install git+https://github.com/tuonglab/dandelion.git
    # or from a specific branch
    pip install git+https://github.com/tuonglab/dandelion@branch_name

Basic requirements
------------------

Python ``>=3.11``

Base packages (auto-installed with ``pip install sc-dandelion``)

.. code::

    numpy>=1.23
    pandas>=1.4
    changeo>=1.1
    anndata>=0.9
    scanpy>=1.9
    scikit-learn>=1.0
    scipy>=1.8
    numba>=0.56
    seaborn>=0.12
    networkx>=3.0
    leidenalg>=0.9
    polyleven>=0.4
    h5py>=3.6
    adjustText>=0.7
    distance>=0.1.3
    plotnine>=0.10
    palettable>=3.3
    mizani>=0.8
    nxviz>=0.7
    rapidfuzz>3.12.1
    zarr>=2.18.7
    circlify>=0.15.0
    airr

Optional extras

.. code:: bash

    pip install sc-dandelion[polars]    # polars>=1.34.0, pyarrow>=21.0.0
    pip install sc-dandelion[scirpy]    # scirpy>=0.21, awkward>=2.1, mudata>=0.2
    pip install sc-dandelion[scrublet]  # scrublet>=0.2, annoy<1.17.0
    pip install sc-dandelion[palantir]  # palantir>=0.2.3, pertpy>=0.1.0, jax>=0.3
    pip install sc-dandelion[dask]      # dask>=2025.11.0, distributed>=2025.11.0, psutil>=6.1.0

Other executables (required for preprocessing)

.. code::

    blast>=2.10.1   (bioconda)
    igblast>=1.15.0 (bioconda)


Acknowledgements
----------------

I would like to acknowledge the contributions from Dr. Chenqu Suo, Dr.
Krysztof Polanksi, Dr. Sarah Teichmann and Prof. Menna Clatworthy, who
helped with the initial conception of the project and for all discussions.

I would also like to acknowledge Dr. Ondrej Suschanek,
Dr. Benjamin Stewart, Dr. Rachel Bashford-Rogers, Dr. Jongeun Park,
Dr. Cecilia-Dominguez Conde, Dr. Kirsten Stewart, Dr. Hamish King and
Dr. Peng He with whom I have had very useful discussions. I would also
like to thank my wife who helped name the package, because she thought
the plots looked like a dandelion =D.

Support
-------

Support is provided on a voluntary basis, as time permits.

If there are any ideas, comments, suggestions, thing you would like to
know more etc., please feel free to email me at z.tuong@uq.edu.au or
post in the issue tracker and I will get back to you.

Citation
--------

Please also cite the following paper if you use version 0.3.0 onwards:

.. [Suo2023] Suo *et al.* (2023),
   *Dandelion uses the single-cell adaptive immune receptor repertoire to explore lymphocyte developmental origins*,
   `Nature Biotechnology <https://www.nature.com/articles/s41587-023-01734-7>`__.

*Chenqu Suo, Krzysztof Polanski, Emma Dann, Rik GH Lindeboom, Roser Vilarrasa-Blasi,
Roser Vento-Tormo, Muzlifah Haniffa, Kerstin B Meyer, Lisa M Dratva,
Zewen Kelvin Tuong, Menna R Clatworthy, Sarah A Teichmann.*
**Dandelion uses single cell adaptive immune receptor repertoire to explore
lymphocyte developmental origins**. Nature Biotechnology 2023.04.13; doi:
https://doi.org/10.1038/s41587-023-01734-7*

The data used in the Nature Biotechnology papers can be found at
`a separate repository <https://github.com/tuonglab/dandelion-demo-files>`__.

``dandelion`` was originally published in:

.. [Stephenson2021] Stephenson *et al.* (2021),
   *Single-cell multi-omics analysis of the immune response in COVID-19*,
   `Nature Medicine <https://www.nature.com/articles/s41591-021-01329-2>`__.

*Emily Stephenson, Gary Reynolds, Rachel A Botting, Fernando J
Calero-Nieto, Michael Morgan, Zewen Kelvin Tuong, Karsten Bach, Waradon
Sungnak, Kaylee B Worlock, Masahiro Yoshida, Natsuhiko Kumasaka,
Katarzyna Kania, Justin Engelbert, Bayanne Olabi, Jarmila Stremenova
Spegarova, Nicola K Wilson, Nicole Mende, Laura Jardine, Louis CS
Gardner, Issac Goh, Dave Horsfall, Jim McGrath, Simone Webb, Michael W
Mather, Rik GH Lindeboom, Emma Dann, Ni Huang, Krzysztof Polanski, Elena
Prigmore, Florian Gothe, Jonathan Scott, Rebecca P Payne, Kenneth F
Baker, Aidan T Hanrath, Ina CD Schim van der Loeff, Andrew S Barr, Amada
Sanchez-Gonzalez, Laura Bergamaschi, Federica Mescia, Josephine L
Barnes, Eliz Kilich, Angus de Wilton, Anita Saigal, Aarash Saleh, Sam M
Janes, Claire M Smith, Nusayhah Gopee, Caroline Wilson, Paul Coupland,
Jonathan M Coxhead, Vladimir Y Kiselev, Stijn van Dongen, Jaume
Bacardit, Hamish W King, Anthony J Rostron, A John Simpson, Sophie
Hambleton, Elisa Laurenti, Paul A Lyons, Kerstin B Meyer, Marko Z
Nikolic, Christopher JA Duncan, Ken Smith, Sarah A Teichmann, Menna R
Clatworthy, John C Marioni, Berthold Gottgens, Muzlifah Haniffa.*
**Single-cell multi-omics analysis of the immune response in
COVID-19**. *Nature Medicine 2021.04.20; doi:
https://dx.doi.org/10.1038/s41591-021-01329-2*


If you use the pre-processing tools/functions, please cite the relevant manuscripts from the immcantation suite, including:

.. [changeo]
*Gupta NT, Vander Heiden JA, Uduman M, Gadala-Maria D, Yaari G, Kleinstein SH.* **Change-O: a toolkit for analyzing large-scale B cell immunoglobulin repertoire sequencing data.** *Bioinformatics 31, 3356-8 (2015). doi: https://doi.org/10.1093/bioinformatics/btv359*

.. [tigger]
*Gadala-Maria D, Yaari G, Uduman M, Kleinstein SH.* **Automated analysis of high-throughput B cell sequencing data reveals a high frequency of novel immunoglobulin V gene segment alleles.** *Proceedings of the National Academy of Sciency of the United States of America, E862-70.*

References
----------

.. [Bashford-Rogers2013] Bashford-Rogers *et al.* (2013),
   *Network properties derived from deep sequencing of human B-cell receptor repertoires delineate B-cell populations*,
   `Genome Research <https://genome.cshlp.org/content/23/11/1874>`__.

.. [Bashford-Rogers2019] Bashford-Rogers *et al.* (2019),
   *Analysis of the B cell receptor repertoire in six immune-mediated diseases*,
   `Nature <https://www.nature.com/articles/s41586-019-1595-3>`__.

.. [Dann2022] Dann *et al.* (2022),
   *Differential abundance testing on single-cell data using k-nearest neighbor graphs*,
   `Nature Biotechnology <https://doi.org/10.1038/s41587-021-01033-z>`__.
   `GitHub <https://github.com/emdann/milopy>`__.

.. [Gadala-Maria2015] Gadala-Maria *et al.* (2015),
   *Automated analysis of high-throughput B cell sequencing data reveals a high frequency of novel immunoglobulin V gene segment alleles*,
   `Proceedings of the National Academy of Sciency of the United States of America <https://www.pnas.org/content/112/8/E862>`__.

.. [Gupta2015] Gupta *et al.* (2015),
   *Change-O: a toolkit for analyzing large-scale B cell immunoglobulin repertoire sequencing data*,
   `Bioinformatics <https://academic.oup.com/bioinformatics/article/31/20/3356/195677>`__.

.. [Irac2024] Irac *et al.* (2024),
   *Single-cell immune repertoire analysis*,
   `Nature Methods <https://www.nature.com/articles/s41592-024-02243-4>`__.

.. [Setty2019] Setty *et al.* (2019)
   *Characterization of cell fate probabilities in single-cell data with Palantir*,
   `Nature Biotechnology <https://doi.org/10.1038/s41587-019-0068-4>`__.
   `GitHub <https://github.com/dpeerlab/Palantir>`__.

.. [Sleckman1998] Sleckman *et al.* (1998)
   *Assembly of productive T cell receptor delta variable region genes exhibits allelic inclusion*,
   `Journal of Experimental Medicine <https://rupress.org/jem/article-lookup/doi/10.1084/jem.188.8.1465>`__.

.. [Stephenson2021] Stephenson *et al.* (2021),
   *Single-cell multi-omics analysis of the immune response in COVID-19*,
   `Nature Medicine <https://www.nature.com/articles/s41591-021-01329-2>`__.

.. [Sturm2020] Sturm *et al.* (2020),
   *Scirpy: a Scanpy extension for analyzing single-cell T-cell receptor-sequencing data*,
   `Bioinformatics <https://academic.oup.com/bioinformatics/article/36/18/4817/5866543>`__.
   `GitHub <https://github.com/icbi-lab/scirpy>`__.

.. [Suo2022] Suo *et al.* (2022),
   *Single cell antigen receptor analysis reveals lymphocyte developmental origins*,
   `bioRxiv <https://doi.org/10.1101/2022.11.18.517068>`__.

.. [Suo2023] Suo *et al.* (2023),
   *Dandelion uses the single-cell adaptive immune receptor repertoire to explore lymphocyte developmental origins*,
   `Nature Biotechnology <https://www.nature.com/articles/s41587-023-01734-7>`__.

.. [Wolf2018] Wolf *et al.* (2018),
   *Scanpy: large-scale single-cell gene expression data analysis*,
   `Genome Biology <https://doi.org/10.1186/s13059-017-1382-0>`__.
   `GitHub <https://github.com/theislab/scanpy>`__.


.. |Docs| image:: https://readthedocs.org/projects/sc-dandelion/badge/?version=latest
   :target: https://sc-dandelion.readthedocs.io/en/latest/?badge=latest
.. |PyPI| image:: https://img.shields.io/pypi/v/sc-dandelion?logo=PyPI
   :target: https://pypi.org/project/sc-dandelion/
.. |Master| image:: https://byob.yarr.is/tuonglab/dandelion/master-version
   :target: https://github.com/tuonglab/dandelion/tree/master
.. |MasterTest| image:: https://github.com/tuonglab/dandelion/actions/workflows/tests.yml/badge.svg?branch=master
   :target: https://github.com/tuonglab/dandelion/actions/workflows/tests.yml
.. |CodeCov| image:: https://codecov.io/gh/tuonglab/dandelion/graph/badge.svg?token=i2BhIz2a4r
   :target: https://codecov.io/gh/tuonglab/dandelion
.. |Colab| image:: https://colab.research.google.com/assets/colab-badge.svg
   :target: https://colab.research.google.com/github/tuonglab/dandelion/blob/master/container/dandelion_singularity.ipynb
.. |logo| image:: notebooks/img/dandelion_logo_illustration.png
.. |overview| image:: notebooks/img/dandelion_overview.png
.. |class| image:: notebooks/img/dandelion_class2.png
