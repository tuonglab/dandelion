Base Backend: ``dandelion.base``
================================

Preprocessing: `pp`
--------------------
.. module:: dandelion.base.preprocessing

.. autosummary::
   :toctree: .

   assign_isotype
   assign_isotypes
   check_contigs
   format_fasta
   format_fastas
   reannotate_genes
   reassign_alleles

Tools: `tl`
-----------
.. module:: dandelion.base.tools

.. autosummary::
   :toctree: .

   clone_centrality
   clone_degree
   clone_diversity
   clone_overlap
   clone_rarefaction
   clone_size
   clone_view
   concat
   define_clones
   extract_edge_weights
   find_clones
   from_scirpy
   generate_network
   productive_ratio
   project_pseudotime_to_cell
   pseudobulk_gex
   pseudotime_transfer
   setup_vdj_pseudobulk
   to_scirpy
   transfer
   vdj_pseudobulk
   vdj_sample
   vj_usage_pca

Plotting: `pl`
--------------
.. module:: dandelion.base.plotting

.. autosummary::
   :toctree: .

   barplot
   clone_circlepackplot
   clone_network
   clone_overlap
   productive_ratio
   spectratype
   stackedbarplot

Reading: `io`
-------------
.. module:: dandelion.base.io

.. autosummary::
   :toctree: .

   read
   read_10x_airr
   read_10x_vdj
   read_airr
   read_bd_airr
   read_parse_airr
   read_seekgene_vdj
   read_ddl
   read_h5ddl

Dandelion Class
---------------
.. currentmodule:: dandelion.base.core

.. autosummary::
   :toctree: .

   Dandelion.add_cell_prefix
   Dandelion.add_cell_suffix
   Dandelion.add_sequence_prefix
   Dandelion.add_sequence_suffix
   Dandelion.compute
   Dandelion.copy
   Dandelion.data
   Dandelion.data_names
   Dandelion.metadata
   Dandelion.metadata_names
   Dandelion.simplify
   Dandelion.store_germline_reference
   Dandelion.update_data
   Dandelion.update_metadata
   Dandelion.update_plus
   Dandelion.write
   Dandelion.write_10x
   Dandelion.write_airr
   Dandelion.write_ddl
   Dandelion.write_h5ddl
   Dandelion.write_vdj


Polars Backend: ``dandelion.polars``
=====================================

Preprocessing: `pp`
--------------------
.. module:: dandelion.polars.preprocessing

.. autosummary::
   :toctree: .

   assign_isotype
   assign_isotypes
   check_contigs
   format_fasta
   format_fastas
   reannotate_genes
   reassign_alleles

Tools: `tl`
-----------
.. module:: dandelion.polars.tools

.. autosummary::
   :toctree: .

   clone_centrality
   clone_degree
   clone_diversity
   clone_overlap
   clone_rarefaction
   clone_size
   clone_view
   concat
   define_clones
   extract_edge_weights
   find_clones
   from_scirpy
   generate_network
   productive_ratio
   project_pseudotime_to_cell
   pseudobulk_gex
   pseudotime_transfer
   setup_vdj_pseudobulk
   to_scirpy
   transfer
   vdj_pseudobulk
   vdj_sample
   vj_usage_pca

Plotting: `pl`
--------------
.. module:: dandelion.polars.plotting

.. autosummary::
   :toctree: .

   barplot
   clone_circlepackplot
   clone_network
   clone_overlap
   productive_ratio
   spectratype
   stackedbarplot

Reading: `io`
-------------
.. module:: dandelion.polars.io

.. autosummary::
   :toctree: .

   read
   read_10x_airr
   read_10x_vdj
   read_airr
   read_bd_airr
   read_parse_airr
   read_seekgene_vdj
   read_ddl
   read_h5ddl
   read_zipddl

Dandelion Class
---------------
.. currentmodule:: dandelion.polars.core

.. autosummary::
   :toctree: .

   Dandelion.add_cell_prefix
   Dandelion.add_cell_suffix
   Dandelion.add_sequence_prefix
   Dandelion.add_sequence_suffix
   Dandelion.clone
   Dandelion.compute
   Dandelion.copy
   Dandelion.data
   Dandelion.data_names
   Dandelion.metadata
   Dandelion.metadata_names
   Dandelion.n_contigs
   Dandelion.n_obs
   Dandelion.reset_ids
   Dandelion.simplify
   Dandelion.store_germline_reference
   Dandelion.to_anndata
   Dandelion.to_eager
   Dandelion.to_lazy
   Dandelion.to_pandas
   Dandelion.to_polars
   Dandelion.update_data
   Dandelion.update_metadata
   Dandelion.update_plus
   Dandelion.write
   Dandelion.write_10x
   Dandelion.write_airr
   Dandelion.write_ddl
   Dandelion.write_h5ddl
   Dandelion.write_vdj
   Dandelion.write_zipddl


Utilities
=========
.. module:: dandelion.utilities

.. autosummary::
   :toctree: .

   extract_edge_weights
   makeblastdb


Tutorial
========
.. module:: dandelion.tutorial

.. autosummary::
   :toctree: .

   setup_dandelion_tutorial_bcr
   setup_dandelion_tutorial_parse
   setup_dandelion_tutorial_tcr
   setup_dandelion_tutorial_trajectory


Logging
=======
.. module:: dandelion.logging

.. autosummary::
   :toctree: .

   print_header
   print_versions

External
========

scanpy
------
.. module:: dandelion.external.scanpy
.. autosummary::
   :toctree: .

   recipe_scanpy_qc

Immcantation
------------

Wrappers for tools in Immcantation pipeline.

Base
~~~~

changeo
^^^^^^^
.. module:: dandelion.external.immcantation.base.changeo

.. autosummary::
   :toctree: .

   assigngenes_igblast
   creategermlines
   makedb_igblast
   parsedb_heavy
   parsedb_light

shazam
^^^^^^
.. module:: dandelion.external.immcantation.base.shazam

.. autosummary::
   :toctree: .

   calculate_threshold
   quantify_mutations

scoper
^^^^^^
.. module:: dandelion.external.immcantation.base.scoper

.. autosummary::
   :toctree: .

   identical_clones
   hierarchical_clones
   spectral_clones

Polars
~~~~~~

changeo
^^^^^^^
.. module:: dandelion.external.immcantation.polars.changeo

.. autosummary::
   :toctree: .

   assigngenes_igblast
   creategermlines
   makedb_igblast
   parsedb_heavy
   parsedb_light

shazam
^^^^^^
.. module:: dandelion.external.immcantation.polars.shazam

.. autosummary::
   :toctree: .

   calculate_threshold
   quantify_mutations

scoper
^^^^^^
.. module:: dandelion.external.immcantation.polars.scoper

.. autosummary::
   :toctree: .

   identical_clones
   hierarchical_clones
   spectral_clones

tigger
~~~~~~
.. module:: dandelion.external.immcantation.tigger

.. autosummary::
   :toctree: .

   tigger_genotype
