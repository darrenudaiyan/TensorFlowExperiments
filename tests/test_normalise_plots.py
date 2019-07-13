import pytest
import numpy as np
from global_functions.normalise_plots import normalize, denormalize

class TestNormalisePlots(object):
    def test_normalise_with_good_array_should_return_correct_result(self):
        
        #arrange
        test_array = np.asarray([1, 2, 3])
        expected_result = np.asarray([-1.22474487,0.,1.22474487])

        #act
        actual_result = normalize(test_array)
        
        #assert
        np.testing.assert_almost_equal(actual_result,expected_result,5)

    def test_denormalise_with_good_array_should_return_correct_result(self):
        #arrange
        test_array = np.asarray([-1.22474487,0.,1.22474487])
        expected_result = np.asarray([-1.22474487,0.,1.22474487])

        #act
        actual_result = denormalize(test_array)
        
        #assert
        np.testing.assert_almost_equal(actual_result,expected_result,5)