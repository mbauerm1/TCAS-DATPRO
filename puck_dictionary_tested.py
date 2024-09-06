import unittest
import numpy as np
from unittest.mock import MagicMock


Pucks = {}

def add_or_update_puck(puck):
    puck_id = puck.get_id()
    
    if not puck.is_alive():
        print(f"Puck with id {puck_id} is not alive. Deleting from the dictionary if exists.")
        delete_puck(puck_id)
        return

    updated_data = {
        'id': puck.get_id(),
        'name': puck.get_name(),
        'position': puck.get_position(),
        'velocity': puck.get_velocity(),
        'acceleration': puck.get_acceleration(),
        'timestamp': puck.get_time(),
        'fuel': puck.get_fuel(),
        'alive': puck.is_alive()
    }

    if puck_id in Pucks:
        Pucks[puck_id].update(updated_data)
    else:
        Pucks[puck_id] = {
            **updated_data,
            'proximity_traffic': False,
            'tca': None,
            'Dtca': None
        }

def delete_puck(puck_id):
    if puck_id in Pucks:
        del Pucks[puck_id]
        

class TestAddOrUpdatePuck(unittest.TestCase):
    def setUp(self):
        
        Pucks.clear()
        
        
        self.puck = MagicMock()
        self.puck.get_id.return_value = 1
        self.puck.get_name.return_value = "TestPuck"
        self.puck.get_position.return_value = np.array([0.0, 0.0])
        self.puck.get_velocity.return_value = np.array([1.0, 1.0])
        self.puck.get_acceleration.return_value = np.array([0.0, 0.0])
        self.puck.get_time.return_value = 123456.789
        self.puck.get_fuel.return_value = 100.0
        self.puck.is_alive.return_value = True

    def test_add_new_puck(self):
        add_or_update_puck(self.puck)
        self.assertIn(1, Pucks)
        self.assertEqual(Pucks[1]['name'], "TestPuck")
        self.assertTrue(np.array_equal(Pucks[1]['position'], np.array([0.0, 0.0])))
        self.assertTrue(np.array_equal(Pucks[1]['velocity'], np.array([1.0, 1.0])))
        self.assertTrue(np.array_equal(Pucks[1]['acceleration'], np.array([0.0, 0.0])))
        self.assertEqual(Pucks[1]['timestamp'], 123456.789)
        self.assertEqual(Pucks[1]['fuel'], 100.0)
        self.assertTrue(Pucks[1]['alive'])
        self.assertFalse(Pucks[1]['proximity_traffic'])
        self.assertIsNone(Pucks[1]['tca'])
        self.assertIsNone(Pucks[1]['Dtca'])

    def test_update_existing_puck(self):
        
        add_or_update_puck(self.puck)
        
        self.puck.get_position.return_value = np.array([1.0, 1.0])
        self.puck.get_velocity.return_value = np.array([2.0, 2.0])
        add_or_update_puck(self.puck)
        self.assertEqual(len(Pucks), 1)
        self.assertEqual(Pucks[1]['name'], "TestPuck")
        self.assertTrue(np.array_equal(Pucks[1]['position'], np.array([1.0, 1.0])))
        self.assertTrue(np.array_equal(Pucks[1]['velocity'], np.array([2.0, 2.0])))
        self.assertTrue(np.array_equal(Pucks[1]['acceleration'], np.array([0.0, 0.0])))
        self.assertEqual(Pucks[1]['timestamp'], 123456.789)
        self.assertEqual(Pucks[1]['fuel'], 100.0)
        self.assertTrue(Pucks[1]['alive'])
        self.assertFalse(Pucks[1]['proximity_traffic'])
        self.assertIsNone(Pucks[1]['tca'])
        self.assertIsNone(Pucks[1]['Dtca'])

    def test_delete_dead_puck(self):
        self.puck.is_alive.return_value = False
        add_or_update_puck(self.puck)
        self.assertNotIn(1, Pucks)

if __name__ == '__main__':
    unittest.main()
