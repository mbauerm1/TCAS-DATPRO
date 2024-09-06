import unittest
import numpy as np


def linear_intersector(vector_function1, vector_function2):
    if vector_function1 is None or vector_function2 is None:
        return None
    
    p1, d1 = vector_function1[:2]  
    p2, d2 = vector_function2[:2]  
    
    dx1, dy1 = d1  
    dx2, dy2 = d2  
    px1, py1 = p1  
    px2, py2 = p2  

    
    denominator = dx1 * dy2 - dy1 * dx2

    if abs(denominator) < 1e-10:  
        return None

    px21 = px2 - px1
    py21 = py2 - py1
    t1 = ((px21) * dy2 - (py21) * dx2) / denominator
    t2 = ((px21) * dy1 - (py21) * dx1) / denominator

    
    if t1 < 0 or t2 < 0:
        return None

    
    intersection_point = np.array([px1 + t1 * dx1, py1 + t1 * dy1])
    return intersection_point, t1, t2


class TestLinearIntersector(unittest.TestCase):
    
    def test_intercect(self):
       
        vector1 = (np.array([0, 0]), np.array([1, 1]))  
        vector2 = (np.array([0, 1]), np.array([1, -1]))  
        result = linear_intersector(vector1, vector2)
        expected_point = np.array([0.5, 0.5]) 
        self.assertIsNotNone(result)
        np.testing.assert_array_almost_equal(result[0], expected_point)
        self.assertAlmostEqual(result[1], 0.5)
        self.assertAlmostEqual(result[2], 0.5)

    def test_parallel(self):
        
        vector1 = (np.array([0, 0]), np.array([1, 1])) 
        vector2 = (np.array([0, 1]), np.array([1, 1]))  
        result = linear_intersector(vector1, vector2)
        self.assertIsNone(result)

    def test_intersect_negative_t(self):
        vector1 = (np.array([0, 0]), np.array([1, 0]))  
        vector2 = (np.array([1, 1]), np.array([0, 1]))  
        result = linear_intersector(vector1, vector2)
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
