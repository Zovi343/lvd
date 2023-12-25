import numpy as np
import pytest
from chromadb.li_index.search.attribtue_filtering.default_filtering import precompute_bucket_ids, compute_ratios_for_attribute_filters, combine_probabilities

def test_precompute_bucket_ids():
    n_categories = [2, 2]
    expected_output = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    assert np.array_equal(precompute_bucket_ids(n_categories), expected_output)

def test_compute_ratios_for_attribute_filters():
    # Setup test data
    n_categories = [2, 2]
    data_prediction = np.array([[0, 0], [0, 1], [1, 0], [0, 1], [1, 1], [0, 0]])
    attribute_filter = np.array([[1, 2, 4, 6]])
    expected_output = [{(0, 0): 0.5, (0, 1): 0.5, (1, 0): 0.0, (1, 1): 0.0}]

    # Expected output might need more complex setup or a known example
    # Here is just a simple check for dictionary structure in the output
    output = compute_ratios_for_attribute_filters(data_prediction, attribute_filter, n_categories)
    assert (output == expected_output)

def test_combine_probabilities():
    # Setup test data
    input_ratios = [{(0, 0): 0.5, (0, 1): 0.5, (1, 0): 0.0, (1, 1): 0.0}]
    expected_output = [{(0, 0): 0.5, (0, 1): 0.5, (1, 0): 0.0, (1, 1): 0.0, (-1, -1): 1.0, (0, -1): 1.0, (1, -1): 0.0}]

    # Perform the test
    output = combine_probabilities(input_ratios)
    print("output", output)
    assert output == expected_output