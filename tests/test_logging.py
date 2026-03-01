#!/usr/bin/env python
import dandelion as ddl


def test_logging():
    """test_logging"""
    ddl.logging.print_header()
    ddl.logging.print_versions()


def test_metadata():
    """test_metadata"""
    assert ddl.__email__ is not None
    assert ddl.__author__ is not None
    assert ddl.__classifiers__ is not None
