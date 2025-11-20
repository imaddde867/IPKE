import sys
sys.path.insert(0, '/projappl/project_2016692/IPKE')

from src.ai.prompting.zero_shot import ZeroShotJSONStrategy
from src.core.unified_config import UnifiedConfig

config = UnifiedConfig.from_environment()
config.prompting_strategy = 'P0'
strategy = ZeroShotJSONStrategy(config)

# Sample chunk with obvious constraints
test_chunk = """
Mix 3M™ Marine Blister Repair Filler and apply. 
Fill 85% of the cavity. 
If needed, feather the repair after 30 minutes.
WARNING: Wear protective gloves.
"""

print("=== PROMPT ===")
print(strategy._format_prompt(strategy.template, "SOP", test_chunk))
