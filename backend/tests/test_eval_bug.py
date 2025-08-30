import unittest
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_generator import AIGenerator
import json


class TestRealEvalBug(unittest.TestCase):
    """Test the actual eval() bug that causes query failures"""
    
    def test_eval_with_json_boolean_fails(self):
        """Test that eval() fails with JSON containing true/false/null"""
        # This is realistic JSON that OpenAI would return
        json_args = '{"query": "Python basics", "enabled": true, "data": null}'
        
        # This should fail
        with self.assertRaises(NameError) as context:
            eval(json_args)
        
        self.assertIn("'true' is not defined", str(context.exception))
    
    def test_json_loads_with_boolean_works(self):
        """Test that json.loads() works correctly"""
        json_args = '{"query": "Python basics", "enabled": true, "data": null}'
        
        # This should work
        result = json.loads(json_args)
        self.assertEqual(result["query"], "Python basics")
        self.assertEqual(result["enabled"], True)
        self.assertIsNone(result["data"])


if __name__ == "__main__":
    unittest.main()