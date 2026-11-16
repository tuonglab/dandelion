"""Compatibility imports for supported Scanpy plotting layouts."""

try:
    from scanpy.plotting.legacy import palettes
    from scanpy.plotting.legacy._tools.scatterplots import embedding
except ModuleNotFoundError as e:
    if e.name != "scanpy.plotting.legacy":
        raise
    from scanpy.plotting import palettes
    from scanpy.plotting._tools.scatterplots import embedding
