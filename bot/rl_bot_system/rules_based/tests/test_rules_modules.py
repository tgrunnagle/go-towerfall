"""
Unit tests for the rules modules (combat, survival, strategic).
"""

import unittest
from unittest.mock import Mock, patch
import math

# Path setup handled by test runner via PYTHONPATH

from rl_bot_system.rules_based.combat_rules import CombatRules
from rl_bot_system.rules_based.survival_rules import SurvivalRules
from rl_bot_system.rules_based.strategic_rules import StrategicRules
from rl_bot_system.rules_based.rules_based_bot import (
    DifficultyLevel,
    ThreatType,
    OpportunityType,
    Threat,
    Opportunity,
    GameStateAnalysis,
    ActionPriority
)


class TestCombatRules(unittest.TestCase):
    """Test cases for the CombatRules class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'accuracy_modifier': 0.8,
            'aggression_level': 0.6,
            'reaction_time': 0.1
        }
        self.combat_rules = CombatRules(self.config)
        
    def test_combat_rules_initialization(self):
        """Test that combat rules initialize correctly"""
        self.assertEqual(self.combat_rules.config, self.config)


class TestSurvivalRules(unittest.TestCase):
    """Test cases for the SurvivalRules class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'accuracy_modifier': 0.8,
            'reaction_time': 0.1
        }
        self.survival_rules = SurvivalRules(self.config)
        
    def test_survival_rules_initialization(self):
        """Test that survival rules initialize correctly"""
        self.assertEqual(self.survival_rules.config, self.config)


class TestStrategicRules(unittest.TestCase):
    """Test cases for the StrategicRules class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = {
            'strategic_depth': 3,
            'aggression_level': 0.6
        }
        self.strategic_rules = StrategicRules(self.config)
        
    def test_strategic_rules_initialization(self):
        """Test that strategic rules initialize correctly"""
        self.assertEqual(self.strategic_rules.config, self.config)


if __name__ == '__main__':
    unittest.main()